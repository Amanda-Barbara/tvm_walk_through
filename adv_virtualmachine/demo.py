import torch
import numpy as np
import torchvision.models as models
import cv2

import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import sys
sys.path.append("/share4algo02/e00130/workspace/tutorial/TensorCompiler/TVM/20230307/tvm/tvm_walk_through")
from utils import array_des
from visualize import RelayVisualizer

def check_optimize(mod,target,params):
  visualizer=RelayVisualizer()
  with tvm.transform.PassContext(opt_level=3):
    compiler = relay.vm.VMCompiler()
    mod,params=compiler.optimize(mod, target=target, params=params)
  print("<Optimized>mod "+str(mod["main"]))
  visualizer.visualize(mod,path="visualizes/memory_opt.prototxt")

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True, progress=True)
  shape_list = [("input0",(1,3,224,224))]
  # fake_input = np.random.random_sample(shape_list[0][1]).astype('float32')
  img_path = "/share4algo02/e00130/workspace/tutorial/TensorCompiler/TVM/20230307/tvm/cat.png"
  img = cv2.imread(img_path).astype("float32")
  img = cv2.resize(img, tuple(shape_list[0][1][2:4]))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = np.transpose(img / 255.0, [2, 0, 1])
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406])
  img /= np.array([0.229, 0.224, 0.225])
  img = np.transpose(img, [2, 0, 1])
  fake_input = np.expand_dims(img, axis=0)
  graph = torch.jit.trace(model,torch.from_numpy(fake_input))

  #step 1 parse to relay
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  target = tvm.target.Target("llvm", host="llvm")
  
  #step 2.1.1 [optional] debug the optimize process
  #check_optimize(mod,target,params)
  
  #step 2 compile the module
  with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

  #step 3 run the VirtualMachine
  dev = tvm.device("llvm", 0)
  vm = VirtualMachine(vm_exec, dev)
  vm.set_input("main", **{"input0": fake_input})
  res=vm.run()
  print("res "+array_des(res))