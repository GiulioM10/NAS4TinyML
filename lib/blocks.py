# GM 05/17/23
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import fvcore.nn as fvc

class Block(Module):
  def __init__(self,
               inchannels: int,
               outchannels: int,
               kernel_size: int,
               downsample: bool = False
               ) -> None:
    """This is the block interface. Each type of block inherits form this class

    Args:
        inchannels (int): Number of input channels
        outchannels (int): Number of output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether this is a downsampling block. Defaults to False.
    """
    super(Block, self).__init__()
    self.inchannels = inchannels
    self.outchannels = outchannels
    self.downsample = downsample
    self.kernel_size = kernel_size
    self.padding = self.kernel_size // 2

    if downsample:
      self.stride = 2
    else:
      self.stride = 1

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    '''
    The forward method to be overloaded by the inheritors
    '''
    pass


class EmptyBlock(Block):
  def __init__(self,
               inchannels: int,
               outchannels: int,
               kernel_size: int,
               downsample=False) -> None:
    """This is an empty block. If it is a downsampling block or the number of channel changes then a 1x1 convolution is 
    applied

    Args:
        inchannels (int): Input channels
        outchannels (int): Output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether to do downsample. Defaults to False.
    """
    super().__init__(inchannels, outchannels, kernel_size, downsample)

    if self.inchannels == self.outchannels and not self.downsample:
      self.cell = nn.Sequential(nn.Identity())
    else:
      self.cell = nn.Sequential(
          nn.Conv2d(
              self.inchannels, self.outchannels,
              kernel_size=1, stride=self.stride
          )
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.cell(x)
  

class MobileNetv2(Block):
  def __init__(self,
               inchannels: int,
               outchannels: int,
               kernel_size :int,
               downsample=False,
               expansion=3) -> None:
    """A state of the art block architecture built to grant good performance
    with a lower number of parameters.
    The expansion term can be changed to vary the size of the inverted 
    bottleneck.

    Args:
        inchannels (int): Input channels
        outchannels (int): Output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether to do downsample. Defaults to False.
        expansion (int, optional): Wether to do downsample. Defaults to 3.
    """
    super().__init__(inchannels, outchannels, kernel_size, downsample)
    # Save expansion hyper-parameter
    self.expansion = expansion
    midchannels = self.outchannels * self.expansion

    # Adapt the tensor
    if self.inchannels == self.outchannels and not self.downsample:
      self.adapt = nn.Sequential(nn.Identity())
    else:
      self.adapt = nn.Sequential(
          nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
          nn.Conv2d(
              self.inchannels, self.outchannels,
              kernel_size=3, stride=self.stride, padding=1
          )
      )

    # Expansion
    self.conv1 = nn.Sequential(
      nn.Conv2d(
          self.outchannels, midchannels,
          kernel_size=1, stride=1
      ),
      nn.BatchNorm2d(midchannels),
      nn.ReLU6()
    )
    
    # Depth-wise convolution layer
    self.depthwise = nn.Sequential(
      nn.Conv2d(
          midchannels, midchannels,
          kernel_size=self.kernel_size, stride=1, padding=self.padding,
          groups=midchannels
      ),
      nn.BatchNorm2d(midchannels),
      nn.ReLU6()
    )

    # Point-wise convolution layer
    self.pointwise = nn.Sequential(
      nn.Conv2d(midchannels, self.outchannels, kernel_size=1),
      nn.BatchNorm2d(self.outchannels)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.adapt(x)
    out = self.conv1(out)
    out = self.depthwise(out)
    out = self.pointwise(out)

    if self.inchannels == self.outchannels and not self.downsample:
      out = out + x

    return out
  

class ConvNext(Block):
  def __init__(self,
               inchannels: int,
               outchannels: int,
               kernel_size: int,
               downsample=False,
               expansion=4) -> None:
    """A state of the art architecture capable of matching the performance
    of trasformers on image classification tasks.

    Args:
        inchannels (int): Input channels
        outchannels (int): Output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether to do downsample. Defaults to False.
        expansion (int, optional): Wether to do downsample. Defaults to 4.
    """
    super().__init__(inchannels, outchannels, kernel_size, downsample)
    # Save expansion hyper-parameter
    self.expansion = expansion
    midchannels = self.outchannels * self.expansion

    # Adapt the tensor
    if self.inchannels == self.outchannels and not self.downsample:
      self.adapt = nn.Sequential(nn.Identity())
    else:
      self.adapt = nn.Sequential(
          nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
          nn.Conv2d(
              self.inchannels, self.outchannels,
              kernel_size=3, stride=self.stride, padding=1
          )
      )

    # ConvNext block
    self.block = nn.Sequential(
        nn.Conv2d(
            self.outchannels, self.outchannels,
            kernel_size=self.kernel_size, stride=1, padding=self.padding,
            groups=self.outchannels
        ),
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
  def __init__(self, inchannels:int,
               outchannels,
               kernel_size: int,
               downsample=False,
               expansion=3) -> None:
    """An Imporved version of the MobileNetv2 block. It uses different activation
    functions and a 'squeeze and excite' layer

    Args:
        inchannels (int): Input channels
        outchannels (int): Output channels
        kernel_size (int): Kernel size
        downsample (bool, optional): Wether to do downsample. Defaults to False.
        expansion (int, optional): Wether to do downsample. Defaults to 3.
    """
    super().__init__(inchannels, outchannels, kernel_size, downsample)

    self.expansion = expansion
    midchannels = self.outchannels * self.expansion

    # Adapt the tensor
    if self.inchannels == self.outchannels and not self.downsample:
      self.adapt = nn.Sequential(nn.Identity())
    else:
      self.adapt = nn.Sequential(
          nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
          nn.Conv2d(
              self.inchannels, self.outchannels,
              kernel_size=3, stride=self.stride, padding=1
          )
      )

    self.block = nn.Sequential(
        nn.Conv2d(self.outchannels, midchannels, kernel_size=1, stride = 1),
        nn.BatchNorm2d(midchannels),
        h_swish(),
        nn.Conv2d(
            midchannels, midchannels,
            kernel_size=self.kernel_size, stride = 1, padding=self.padding,
            groups = midchannels
        ),
        nn.BatchNorm2d(midchannels),
        SELayer(midchannels),
        h_swish(),
        nn.Conv2d(midchannels, self.outchannels, kernel_size=1, stride=1),
        nn.BatchNorm2d(self.outchannels)
    )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.adapt(x)
    out = self.block(out)

    if self.inchannels == self.outchannels and not self.downsample:
      out = out + x

    return out
  

### Unused Blocks
# class ResNet(Block):
#   def __init__(self,
#                inchannels: int,
#                outchannels: int,
#                downsample=False
#                ) -> None:
#     super().__init__(inchannels, outchannels, downsample)

#     '''
#     This class implements the classic Residual block.
#     If the input is downsampled or the number of features changes the shortcut
#     adapts the original tensor accordingly
#     '''

#     ## The residual Block
#     self.cell = nn.Sequential(
#         nn.Conv2d(
#             self.inchannels, self.outchannels,
#             kernel_size=3, stride=self.stride, padding=1
#         ),
#         nn.BatchNorm2d(self.outchannels),
#         nn.ReLU(),
#         nn.Conv2d(
#             self.outchannels, self.outchannels,
#             kernel_size=3, stride=1, padding=1
#         ),
#         nn.BatchNorm2d(self.outchannels)
#     ) 

#     ## The shortcut
#     if self.inchannels == self.outchannels and not self.downsample:
#       self.shortcut = nn.Sequential(nn.Identity())
#     else:
#       self.shortcut = nn.Sequential(
#           nn.Conv2d(
#               self.inchannels, self.outchannels,
#               kernel_size=1, stride=self.stride
#           ),
#           nn.BatchNorm2d(self.outchannels)
#       )

#     ## Final activation
#     self.act = nn.ReLU()

#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     F = self.cell(x) # Residual
#     x = self.shortcut(x) # Adjust the dimensions of the tensor
#     out = F + x # Sum residual and original value
#     out = self.act(out) # Non-linear activation function

#     return out


# class ImpResNet(Block):
#   def __init__(self,
#                inchannels: int,
#                outchannels: int,
#                kernel_size:int,
#                downsample=False) -> None:
#     super().__init__(inchannels, outchannels, kernel_size, downsample)

#     '''
#     This class implements an improved version of the residual block.
#     If the input is downsampled or the number of features changes the shortcut
#     adapts the original tensor accordingly
#     '''

#     if self.inchannels == self.outchannels and not self.downsample:
#       self.adapt = nn.Sequential(nn.Identity())
#     else:
#       self.adapt = nn.Sequential(
#           nn.GroupNorm(num_groups=1, num_channels=self.inchannels),
#           nn.Conv2d(
#               self.inchannels, self.outchannels,
#               kernel_size=3, stride=self.stride, padding=1
#           )
#       )


#     # Residual block
#     self.cell = nn.Sequential(
#         nn.BatchNorm2d(self.inchannels),
#         nn.ReLU(),
#         nn.Conv2d(
#             self.outchannels, self.outchannels,
#             kernel_size=self.kernel_size, stride=1, padding=self.padding
#         ),
#         nn.BatchNorm2d(self.outchannels),
#         nn.ReLU(),
#         nn.Conv2d(
#             self.outchannels, self.outchannels,
#             kernel_size=self.kernel_size, stride=1, padding=self.padding
#         )
#     )


#     # Shortcut
#     if self.inchannels == self.outchannels and not self.downsample:
#       self.shortcut = nn.Sequential()
#     else:
#       self.shortcut = nn.Sequential(
#           nn.BatchNorm2d(self.inchannels),
#           nn.ReLU(),
#           nn.Conv2d(
#               self.inchannels, self.outchannels,
#               kernel_size=1, stride=self.stride
#           )
#       )
  
#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     F = self.cell(x)
#     x = self.shortcut(x)
#     out = F + x

#     return out