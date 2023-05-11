from gelNetwork import GELnetwork
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import fvcore.nn as fvc
import types
from typing import List, Dict

class ISpace:
  def __init__(self) -> None:
    pass

class Individual:
  def __init__(self, genotype: List[List], space: ISpace, device: torch.device, generation = 0) -> None:
    """The object containing all the info about a certain architecture. 

    Args:
        genotype (List[List]): The genomic sequence of the individual
        space (ISpace): The space to which the individual belongs
        device (torch.device): The device on wich the individual's network will be loaded to
        generation (int, optional): The generation of the individual. Defaults to 0.
    """
    super().__init__()
    self.genotype = genotype
    self.generation = generation
    self.space = space
    self.device = device
    self.born = False
    self.has_metrics = False

    self._cost_info = None # dict{n. param, n. flops}
    self._metrics = None #
    self.rank = None

  def get_network(self) -> Module:
    """Returns a torch.nn.Module object built usic the genome

    Returns:
        Module: A neural network built according to the individual genes
    """
    net = GELnetwork(self.genotype)

    if self.device is not None:
      net = net.to(self.device)
    
    return net

  def _compute_cost_info(self):
    """Compute and save the individuals costs in terms of number of parameters (milions) and flops (milions)
    """
    net = self.get_network()
    costs = {"params": None, "flops": None}
    costs["params"] = sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6
    costs["flops"] = fvc.FlopCountAnalysis(net, torch.rand((1, 3, 112, 112)).to(self.device))\
    .unsupported_ops_warnings(False)\
    .uncalled_modules_warnings(False)\
    .total()/1e6
    self._cost_info = costs

  def set_metrics(self, metrics: Dict = None):
    """COmpute or set an individual's metrics

    Args:
        metrics (Dict, optional): Metrics dictionary or None. If None the metrics are computed and
        then saved. Defaults to None.
    """
    if metrics is None:
      self._metrics = self.space.compute_tfm(self)
    else:
      self._metrics = metrics
    self.has_metrics = True
  
  def get_metrics(self) -> Dict:
    """Get indivisual's metrics

    Returns:
        Dict: The individual's metrics
    """
    if not self.has_metrics:
      self.set_metrics()
    return self._metrics
  
  def set_cost_info(self, cost_info: Dict = None):
    """Use this method to set an individual's cost info if they are already known.

    Args:
        cost_info (Dict, optional): Cost Info dictionary or None. If None the info are computed and
        then saved. Defaults to None.
    """
    if cost_info is None:
      self._compute_cost_info()
    else:
      self._cost_info = cost_info

  def get_cost_info(self) -> Dict:
    """Get indivisual's cost info

    Returns:
        Dict: The individual's cost info
    """
    if self._cost_info is not None:
      return self._cost_info
    else:
      self._compute_cost_info()
      return self._cost_info
    
  def set_generation(self, gen: int):
    """Set individual's generation

    Args:
        gen (int): The generation of the individual
    """
    self.generation = gen

  def print_structure(self):
    """Print on terminal the individual structure
    """
    print("---- INDIVIDUAL STRUCTURE ---")
    print("Stem Block")
    for index, gene in enumerate(self.genotype):
      blocktype, ks, chan, expa, dwns = gene
      print("Block {} : {}(Kernel = {}, Outchannles = {}, Expansion = {}, Downsample = {})".format(index + 1, blocktype, ks, chan, expa, dwns))
    print("GlobalAVG Pool")
    print("Final Layer")