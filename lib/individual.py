from gelNetwork import GELnetwork
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import fvcore.nn as fvc
import types
from typing import List

class ISpace:
  def __init__(self) -> None:
    pass

class Individual:
  def __init__(self, genotype: List[List], space: ISpace, device: torch.device, generation = 0) -> None:
    super().__init__()

    self.genotype = genotype
    self.generation = generation
    self.space = space
    self.device = device
    self.born = False
    self.has_metrics = False

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
    costs["flops"] = fvc.FlopCountAnalysis(net, torch.rand((1, 3, 112, 112)).to(self.device))\
    .unsupported_ops_warnings(False)\
    .uncalled_modules_warnings(False)\
    .total()/1e6
    self._cost_info = costs

  def set_metrics(self, metrics: dict = None):
    if metrics is None:
      self._metrics = self.space.compute_tfm(self)
    else:
      self._metrics = metrics
    self.has_metrics = True
  
  def get_metrics(self) -> dict:
    if not self.has_metrics:
      self.set_metrics()
    return self._metrics
  
  def set_cost_info(self, cost_info: dict = None):
    if cost_info is None:
      self._compute_cost_info
    else:
      self._cost_info = cost_info

  def get_cost_info(self) -> dict:
    if self._cost_info is not None:
      return self._cost_info
    else:
      self._compute_cost_info(self.get_network())
      return self._cost_info
    
  def set_generation(self, gen: int):
    self.generation = gen

  def print_structure(self):
    print("---- INDIVIDUAL STRUCTURE ---")
    print("Stem Block")
    for index, gene in enumerate(self.genotype):
      blocktype, ks, chan, expa, dwns = gene
      print("Block {} : {}(Kernel = {}, Outchannles = {}, Expansion = {}, Downsample = {})".format(index + 1, blocktype, ks, chan, expa, dwns))
    print("GlobalAVG Pool")
    print("Final Layer")