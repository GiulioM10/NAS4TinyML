import torch
from torch.nn import Module
import torch.nn as nn


class Block(Module):
  def __init__(self, inchannels, outchannels, downsample = False) -> None:
    super(Block, self).__init__()
    self.inchannels = inchannels
    self.outchannels = outchannels
    self.downsample = downsample
    if downsample:
      self.stride = 2
    else:
      self.stride = 1

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    pass

class ResNet(Block):
  def __init__(self, inchannels, outchannels, downsample = False) -> None:
    super().__init__(inchannels, outchannels, downsample)

    self.cell = nn.Sequential(
        nn.Conv2d(self.inchannels, self.outchannels, kernel_size=3, stride=self.stride, padding=1),
        nn.BatchNorm2d(self.outchannels),
        nn.ReLU(),
        nn.Conv2d(self.outchannels, self.outchannels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(self.outchannels)
    )

    if self.inchannels == self.outchannels and not self.downsample:
      self.shortcut = nn.Sequential()
    else:
      self.shortcut = nn.Sequential(
          nn.Conv2d(self.inchannels, self.outchannels, kernel_size=1, stride=self.stride),
          nn.BatchNorm2d(self.outchannels)
      )

    self.act = nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    F = self.cell(x)
    x = self.shortcut(x)
    out = F + x
    out = self.act(out)

    return out
  
class ImpResNet(Block):
  def __init__(self, inchannels, outchannels, downsample = False) -> None:
    super().__init__(inchannels, outchannels, downsample)

    self.cell = nn.Sequential(
        nn.BatchNorm2d(self.inchannels),
        nn.ReLU(),
        nn.Conv2d(self.inchannels, self.outchannels, kernel_size=3, stride=self.stride, padding=1),
        nn.BatchNorm2d(self.outchannels),
        nn.ReLU(),
        nn.Conv2d(self.outchannels, self.outchannels, kernel_size=3, stride=1, padding=1)
    )

    if self.inchannels == self.outchannels and not self.downsample:
      self.shortcut = nn.Sequential()
    else:
      self.shortcut = nn.Sequential(
          nn.BatchNorm2d(self.inchannels),
          nn.ReLU(),
          nn.Conv2d(self.inchannels, self.outchannels, kernel_size=1, stride=self.stride)
      )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    F = self.cell(x)
    x = self.shortcut(x)
    out = F + x

    return out

class EmptyBlock(Block):
  def __init__(self, inchannels, outchannels, downsample=False) -> None:
    super().__init__(inchannels, outchannels, downsample)

    if self.inchannels == self.outchannels and not self.downsample:
      self.cell = nn.Sequential(nn.Identity())
    else:
      self.cell = nn.Sequential(
          nn.Conv2d(self.inchannels, self.outchannels, kernel_size=1, stride=self.stride)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.cell(x)
  
class MobileNetv2(Block):
  def __init__(self, inchannels, outchannels, downsample=False, expansion = 3) -> None:
    super().__init__(inchannels, outchannels, downsample)

    self.expansion = expansion
    midchannels = self.inchannels * self.expansion


    if not(self.expansion == 1):
      self.conv1 = nn.Sequential(
        nn.Conv2d(self.inchannels, midchannels, kernel_size=1, stride=1),
        nn.BatchNorm2d(midchannels),
        nn.ReLU6()
      )
    
    self.depthwise = nn.Sequential(
      nn.Conv2d(midchannels, midchannels, kernel_size=3, stride=self.stride, padding=1, groups=midchannels),
      nn.BatchNorm2d(midchannels),
      nn.ReLU6()
    )

    self.pointwise = nn.Sequential(
      nn.Conv2d(midchannels, self.outchannels, kernel_size=1),
      nn.BatchNorm2d(self.outchannels)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.expansion == 1:
      out = x
    else:
      out = self.conv1(x)
    out = self.depthwise(out)
    out = self.pointwise(out)

    if self.inchannels == self.outchannels and not self.downsample:
      out = out + x

    return out

class ConvNext(Block):
  def __init__(self, inchannels, outchannels, downsample=False, expansion = 4) -> None:
    super().__init__(inchannels, outchannels, downsample)

    self.expansion = expansion
    midchannels = self.outchannels * self.expansion

    if self.inchannels == self.outchannels and not self.downsample:
      self.adapt = nn.Sequential(nn.Identity())
    else:
      self.adapt = nn.Sequential(
          nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
          nn.Conv2d(self.inchannels, self.outchannels, kernel_size=3, stride=self.stride, padding=1)
      )

    self.block = nn.Sequential(
        nn.Conv2d(self.outchannels, self.outchannels, kernel_size=7, stride=1, padding=3, groups=self.outchannels),
        nn.GroupNorm(num_groups=1, num_channels=self.outchannels),
        nn.Conv2d(self.outchannels, midchannels, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(midchannels, self.outchannels, kernel_size=1)
    )


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.adapt(x)
    F = self.block(x)
    out = x + F
    
    return out
  
class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction),
                nn.ReLU(),
                nn.Linear(channel//reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class MobileNetv3(Block):
  def __init__(self, inchannels, outchannels, downsample=False, expansion = 3) -> None:
    super().__init__(inchannels, outchannels, downsample)

    self.expansion = expansion
    midchannels = self.inchannels * self.expansion

    self.block = nn.Sequential(
        nn.Conv2d(self.inchannels, midchannels, kernel_size=1, stride = 1),
        nn.BatchNorm2d(midchannels),
        h_swish(),
        nn.Conv2d(midchannels, midchannels, kernel_size=3, stride = self.stride, padding=1, groups = midchannels),
        nn.BatchNorm2d(midchannels),
        SELayer(midchannels),
        h_swish(),
        nn.Conv2d(midchannels, self.outchannels, kernel_size=1, stride=1),
        nn.BatchNorm2d(self.outchannels)
    )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.block(x)

    if self.inchannels == self.outchannels and not self.downsample:
      out = out + x

    return out
  
def block_builder(block_type, inchannels, downsample):

  # genelist = [
  #   "Empty",
  #   "ResNet_64",
  #   "ResNet_128",
  #   "ResNet_256",
  #   "ResNet_512",
  #   "ImpResNet_64",
  #   "ImpResNet_128",
  #   "ImpResNet_256",
  #   "ImpResNet_512",
  #   "MobileNetv2_64",
  #   "MobileNetv2_128",
  #   "MobileNetv2_256",
  #   "MobileNetv2_512",
  #   "MobileNetv3_64",
  #   "MobileNetv3_128",
  #   "MobileNetv3_256",
  #   "MobileNetv3_512",
  #   "ConvNext_64",
  #   "ConvNext_128",
  #   "ConvNext_256",
  #   "ConvNext_512",
  # ]

  block_type = genelist[block_type]

  if block_type == 0:
      block = EmptyBlock(inchannels=inchannels, outchannels=inchannels, downsample=downsample)
  elif block_type == 1: # ResNet 64
    block = ResNet(inchannels=inchannels, outchannels=64, downsample=downsample)
  elif block_type == 2: # ResNet 128
    block = ResNet(inchannels=inchannels, outchannels=128, downsample=downsample)
  # elif block_type == "ResNet_256": # ResNet 256
  #   block = ResNet(inchannels=inchannels, outchannels=256, downsample=downsample)
  # elif block_type == "ResNet_512": # ResNet 512
  #   block = ResNet(inchannels=inchannels, outchannels=512, downsample=downsample)
  # elif block_type == "ImpResNet_64": # ImpResNet 64
  #   block = ImpResNet(inchannels=inchannels, outchannels=64, downsample=downsample)
  # elif block_type == "ImpResNet_128": # ImpResNet 128
  #   block = ImpResNet(inchannels=inchannels, outchannels=128, downsample=downsample)
  # elif block_type == "ImpResNet_256": # ImpResNet 256
  #   block = ImpResNet(inchannels=inchannels, outchannels=256, downsample=downsample)
  # elif block_type == "ImpResNet_512": # ImpResNet 512
  #   block = ImpResNet(inchannels=inchannels, outchannels=512, downsample=downsample)
  # elif block_type == "MobileNetv2_64":
  #   block = MobileNetv2(inchannels=inchannels, outchannels=64, downsample=downsample)
  # elif block_type == "MobileNetv2_128":
  #   block = MobileNetv2(inchannels=inchannels, outchannels=128, downsample=downsample)
  # elif block_type == "MobileNetv2_256":
  #   block = MobileNetv2(inchannels=inchannels, outchannels=256, downsample=downsample)
  # elif block_type == "MobileNetv2_512":
  #   block = MobileNetv2(inchannels=inchannels, outchannels=512, downsample=downsample)
  # elif block_type == "MobileNetv3_64":
  #   block = MobileNetv3(inchannels=inchannels, outchannels=64, downsample=downsample)
  # elif block_type == "MobileNetv3_128":
  #   block = MobileNetv3(inchannels=inchannels, outchannels=128, downsample=downsample)
  # elif block_type == "MobileNetv3_256":
  #   block = MobileNetv3(inchannels=inchannels, outchannels=256, downsample=downsample)
  # elif block_type == "MobileNetv3_512":
  #   block = MobileNetv3(inchannels=inchannels, outchannels=512, downsample=downsample)
  # elif block_type == "ConvNext_64":
  #   block = ConvNext(inchannels=inchannels, outchannels=64, downsample=downsample)
  # elif block_type == "ConvNext_128":
  #   block = ConvNext(inchannels=inchannels, outchannels=128, downsample=downsample)
  # elif block_type == "ConvNext_256":
  #   block = ConvNext(inchannels=inchannels, outchannels=256, downsample=downsample)
  # elif block_type == "ConvNext_512":
  #   block = ConvNext(inchannels=inchannels, outchannels=512, downsample=downsample)
  return block, block.outchannels

class GELnetwork(Module):
  def __init__(self, genome, *args, **kwargs) -> None:
    super(GELnetwork, self).__init__(*args, **kwargs)
    self.blocks = nn.ModuleList([])

    stem_conv = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU()
    ) # 64 x 56 x 56 tensor

    self.blocks.append(stem_conv)

    inchannels = 64
    block = None

    for index, gene in enumerate(genome):
      if index%4 == 3:
        block, inchannels = block_builder(gene, inchannels, downsample=True)
      else:
        block, inchannels = block_builder(gene, inchannels, downsample=False)
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