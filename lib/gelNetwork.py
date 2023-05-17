# GM 05/17/23
from blocks import Block, MobileNetv2, MobileNetv3, ConvNext, EmptyBlock
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import fvcore.nn as fvc
import types
from typing import List

def block_builder(block: str, ks: int, inchan: int, outchan: int, expa: int, dwns: bool) -> Block:
  """This function returns a Block given its description in terms of type, kernel_size, channels, 
  expansion factor, and wether the block performs downsample

  Args:
      block (str): The name of the block type
      ks (int): Kernel_size
      inchan (int): Input channels
      outchan (int): Output channels
      expa (int): Expansion factor
      dwns (bool): Wether to perform downsample

  Raises:
      ValueError: The block is not implemented in blocks.py

  Returns:
      Block: An appropriate instance of the block class
  """
  if block == "MobileNetv2":
    block = MobileNetv2(
        inchannels=inchan,
        outchannels=outchan,
        kernel_size=ks,
        expansion=expa,
        downsample=dwns)
  elif block == "MobileNetv3":
    block = MobileNetv3(
        inchannels=inchan,
        outchannels=outchan,
        kernel_size=ks,
        expansion=expa,
        downsample=dwns)
  elif block == "ConvNext":
    block = ConvNext(
        inchannels=inchan,
        outchannels=outchan,
        kernel_size=ks,
        expansion=expa,
        downsample=dwns)
  else:
    raise ValueError("Unknown block")
  return block


class GELnetwork(Module):
  def __init__(self, genome: List[List],*args, **kwargs) -> None:
    """This is the net wrapper for the GELSearchSpace

    Args:
        genome (List[List]): The genome containing the network structure
    """
    super(GELnetwork, self).__init__(*args, **kwargs)
    self.blocks = nn.ModuleList([])


    # FARE PROVE USANDO SIA UNA CONV2D CHE L'ALTRA
    # stem_conv = nn.Sequential(
    #   # nn.Conv2d(3, 64, kernel_size=5, stride=4, padding=2), # 64 x 28 x 28 tensor
    #   nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64 x 56 x 56 tensor
    #   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    #   nn.BatchNorm2d(64),
    #   nn.ReLU()
    # ) # 64 x 56 x 56 tensor


    # SOSTITUIRE IL BLOCCO INIZIALE CON IL SEGUENTE
    stem_conv = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0),  # input 3 x 112 x 112
    nn.BatchNorm2d(64),
    nn.ReLU()
    ) # 64 x 28 x 28 tensor

    self.blocks.append(stem_conv)

    inchannels = 64

    for index, gene in enumerate(genome):
      blocktype, ks, chan, expa, dwns = gene
      block = block_builder(blocktype, ks, inchannels, chan, expa, dwns)
      inchannels = chan
      self.blocks.append(block)

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(inchannels, 2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      x = block(x)
    x = self.gap(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x