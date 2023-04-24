import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import fvcore.nn as fvc
from Blocks import *

class Individual:
  def __init__(self, genotype, device, generation = 0) -> None:
    super().__init__()

    self.genotype = genotype
    self.generation = generation
    self.device = device
    self.born = False

    self._cost_info = None # dict{n. param, n. flops}
    self._metrics = None # 

  def get_network(self) -> Module:
    net = GELnetwork(self.genotype)

    if self.device is not None:
      net = net.to(self.device)
    
    return net

  def _compute_cost_info(self, net: Module):
    costs = {"params": None, "flops": None}
    costs["params"] = sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6
    costs["flops"] = fvc.FlopCountAnalysis(net, torch.rand((1, 3, 224, 224)).to(self.device)).unsupported_ops_warnings(False).total()/1e6

    self._cost_info = costs

  def get_cost_info(self):
    if self._cost_info is not None:
      return self._cost_info
    else:
      self._compute_cost_info(self.get_network())
      return self._cost_info