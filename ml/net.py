import mlx.nn as nn
import mlx.core as mx

#NOTE: input as B,W,H,C
class Net(nn.Module):
  def __init__(self, blocks, size, in_features, filters=64):
    super().__init__()
    self.resblocks = [ResBlock(filters) for _ in range(blocks)] 
    self.convblock = ConvBlock(in_features)
    self.vnet = ValueNet(size, filters) 
    self.qnet = PolicyNet(size, filters)
  
  def __call__(self,x):
    x = self.convblock(x)
    for resblock in self.resblocks:
      x = resblock(x)
    return self.qnet(x), self.vnet(x)

class ConvBlock(nn.Module):
  def __init__(self, in_features, filters=64):
    super().__init__()
    self.conv = nn.Conv2d(in_features, filters, kernel_size=3, stride=1, padding=1)
    self.norm = nn.GroupNorm(8,filters)
  
  def __call__(self,x):
    x = self.conv(x)
    return mx.maximum(self.norm(x),0.0,stream=mx.cpu)

class ResBlock(nn.Module):
  def __init__(self, filters):
    super().__init__()
    self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
    self.norm1 = nn.GroupNorm(8,filters)
    self.norm2 = nn.GroupNorm(8,filters)
  
  def __call__(self,x):
    x1 = self.norm1(self.conv1(x))
    x2 = mx.maximum(x1,0.0, stream=mx.cpu)
    x3 = self.norm2(self.conv2(x2))
    return mx.maximum(x3+x,0.0, stream=mx.cpu)

class PolicyNet(nn.Module):
  def __init__(self, size, in_features):
    super().__init__() 
    self.conv = nn.Conv2d(in_features, 2, kernel_size=3, stride=1, padding=1) 
    self.norm = nn.GroupNorm(1,2)
    self.linear = nn.Linear(2*size**2,size**2+1)
  
  def __call__(self,x):
    x = self.conv(x)
    x = self.norm(x)
    x = mx.maximum(x,0.0, stream=mx.cpu)
    x = mx.reshape(x, [x.shape[0],2*x.shape[1]**2], stream=mx.cpu)
    return mx.softmax(self.linear(x), stream=mx.cpu)

class ValueNet(nn.Module):
  def __init__(self, size, in_features):
    super().__init__() 
    self.conv = nn.Conv2d(in_features, 1, kernel_size=1, stride=1) 
    self.norm = nn.GroupNorm(1,1)
    self.linear1 = nn.Linear(size**2,128)    
    self.linear2 = nn.Linear(128,1)
  
  def __call__(self,x):
    x = self.conv(x)
    x = self.norm(x)
    x = mx.maximum(x,0.0, stream=mx.cpu)
    x = mx.reshape(x, [x.shape[0],x.shape[1]**2], stream=mx.cpu)
    x = self.linear1(x)
    x = mx.maximum(x,0.0, stream=mx.cpu)
    x = self.linear2(x)
    return mx.tanh(x, stream=mx.cpu)

    


