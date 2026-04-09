"""ERes2Net code"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import training_ssondo.utils.student_models.ERes2Net.pooling_layers as pooling_layers
from training_ssondo.utils.student_models.utils import count_parameters


class AFF(nn.Module):

  def __init__(self, channels=64, r=4):
    super(AFF, self).__init__()
    inter_channels = int(channels // r)

    self.local_att = nn.Sequential(
        nn.Conv2d(channels * 2, inter_channels,
                  kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(inter_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(inter_channels, channels,
                  kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(channels),
    )

  def forward(self, x, ds_y):
    xa = torch.cat((x, ds_y), dim=1)
    x_att = self.local_att(xa)
    x_att = 1.0 + torch.tanh(x_att)
    xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)

    return xo


class ReLU(nn.Hardtanh):

  def __init__(self, inplace=False):
    super(ReLU, self).__init__(0, 20, inplace)

  def __repr__(self):
    inplace_str = 'inplace' if self.inplace else ''
    return self.__class__.__name__ + ' (' \
        + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
  "1x1 convolution without padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                   padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlockERes2Net(nn.Module):
  expansion = 2

  def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
    super(BasicBlockERes2Net, self).__init__()
    width = int(math.floor(planes * (baseWidth / 64.0)))
    self.conv1 = conv1x1(in_planes, width * scale, stride)
    self.bn1 = nn.BatchNorm2d(width * scale)
    self.nums = scale

    convs = []
    bns = []
    for i in range(self.nums):
      convs.append(conv3x3(width, width))
      bns.append(nn.BatchNorm2d(width))
    self.convs = nn.ModuleList(convs)
    self.bns = nn.ModuleList(bns)
    self.relu = ReLU(inplace=True)

    self.conv3 = conv1x1(width * scale, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
          nn.BatchNorm2d(self.expansion * planes))
    self.stride = stride
    self.width = width
    self.scale = scale

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    spx = torch.split(out, self.width, 1)
    for i in range(self.nums):
      if i == 0:
        sp = spx[i]
      else:
        sp = sp + spx[i]
      sp = self.convs[i](sp)
      sp = self.relu(self.bns[i](sp))
      if i == 0:
        out = sp
      else:
        out = torch.cat((out, sp), 1)

    out = self.conv3(out)
    out = self.bn3(out)

    residual = self.shortcut(x)
    out += residual
    out = self.relu(out)

    return out


class BasicBlockERes2Net_diff_AFF(nn.Module):
  expansion = 2

  def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
    super(BasicBlockERes2Net_diff_AFF, self).__init__()
    width = int(math.floor(planes * (baseWidth / 64.0)))
    self.conv1 = conv1x1(in_planes, width * scale, stride)
    self.bn1 = nn.BatchNorm2d(width * scale)
    self.nums = scale

    convs = []
    fuse_models = []
    bns = []
    for i in range(self.nums):
      convs.append(conv3x3(width, width))
      bns.append(nn.BatchNorm2d(width))
    for j in range(self.nums - 1):
      fuse_models.append(AFF(channels=width))

    self.convs = nn.ModuleList(convs)
    self.bns = nn.ModuleList(bns)
    self.fuse_models = nn.ModuleList(fuse_models)
    self.relu = ReLU(inplace=True)

    self.conv3 = conv1x1(width * scale, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
          nn.BatchNorm2d(self.expansion * planes))
    self.stride = stride
    self.width = width
    self.scale = scale

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    spx = torch.split(out, self.width, 1)
    for i in range(self.nums):
      if i == 0:
        sp = spx[i]
      else:
        sp = self.fuse_models[i - 1](sp, spx[i])

      sp = self.convs[i](sp)
      sp = self.relu(self.bns[i](sp))
      if i == 0:
        out = sp
      else:
        out = torch.cat((out, sp), 1)

    out = self.conv3(out)
    out = self.bn3(out)

    residual = self.shortcut(x)
    out += residual
    out = self.relu(out)

    return out


class ERes2Net(nn.Module):
  def __init__(self,
               block=BasicBlockERes2Net,
               block_fuse=BasicBlockERes2Net_diff_AFF,
               num_blocks=[3, 4, 6, 3],
               m_channels=32,
               feat_dim=80,
               pooling_func='TSTP',
               add_layer=False):
    super(ERes2Net, self).__init__()
    self.add_layer = add_layer
    self.in_planes = m_channels
    self.feat_dim = feat_dim

    if self.add_layer:
      self.stats_dim = int(feat_dim / 16) * m_channels * 2
    else:
      self.stats_dim = int(feat_dim / 8) * m_channels * 8

    self.conv1 = nn.Conv2d(1,
                           m_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(m_channels)
    self.layer1 = self._make_layer(block,
                                   m_channels,
                                   num_blocks[0],
                                   stride=1)
    self.layer2 = self._make_layer(block,
                                   m_channels * 2,
                                   num_blocks[1],
                                   stride=2)
    self.layer3 = self._make_layer(block_fuse,
                                   m_channels * 4,
                                   num_blocks[2],
                                   stride=2)
    self.layer4 = self._make_layer(block_fuse,
                                   m_channels * 8,
                                   num_blocks[3],
                                   stride=2)
    if self.add_layer:
      self.layer5 = self._make_layer(block_fuse,
                                     m_channels * 2,
                                     num_blocks[4],
                                     stride=2)

    # Downsampling module for each layer
    self.layer1_downsample = nn.Conv2d(
        m_channels * 2,
        m_channels * 4,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False)
    self.layer2_downsample = nn.Conv2d(
        m_channels * 4,
        m_channels * 8,
        kernel_size=3,
        padding=1,
        stride=2,
        bias=False)
    self.layer3_downsample = nn.Conv2d(
        m_channels * 8,
        m_channels * 16,
        kernel_size=3,
        padding=1,
        stride=2,
        bias=False)
    if self.add_layer:
      self.layer4_downsample = nn.Conv2d(
          m_channels * 16,
          m_channels * 4,
          kernel_size=3,
          padding=1,
          stride=2,
          bias=False)

    # Bottom-up fusion module
    self.fuse_mode12 = AFF(channels=m_channels * 4)
    self.fuse_mode123 = AFF(channels=m_channels * 8)
    self.fuse_mode1234 = AFF(channels=m_channels * 16)
    if self.add_layer:
      self.fuse_mode12345 = AFF(channels=m_channels * 4)

    self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
    self.pool = getattr(pooling_layers, pooling_func)(
        in_dim=self.stats_dim * block.expansion)

    self.n_parameters = count_parameters(self)
    print("Number of trainable parameters - ERes2Net: ", self.n_parameters)

    self.emb_size = self.stats_dim * block.expansion * self.n_stats
    print("Size of the embedding: ", self.emb_size)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    # x shape (batch, time_steps, frequency_frames, time_frames)
    if x.ndim < 3:
      x = x.unsqueeze_(1)
    # Reshaping x into batch * time_steps, 1, frequency_frames, time_frames)
    b, ts, f, tf = x.shape
    x = x.reshape(b * ts, 1, f, tf)

    x = F.relu(self.bn1(self.conv1(x)))

    # Out1
    x = self.layer1(x)

    # Out2
    out2 = self.layer2(x)

    # Layer1 and Layer2 Fuse
    x = self.layer1_downsample(x)
    fuse_out12 = self.fuse_mode12(out2, x)

    # Out3
    x = self.layer3(out2)

    # Layer12 and layer3 Fuse
    fuse_out12_downsample = self.layer2_downsample(fuse_out12)
    fuse_out123 = self.fuse_mode123(x, fuse_out12_downsample)

    # Out4
    out2 = self.layer4(x)

    # Layer123 and layer4 Fuse
    fuse_out123_downsample = self.layer3_downsample(fuse_out123)
    fuse_out1234 = self.fuse_mode1234(out2, fuse_out123_downsample)

    if self.add_layer:
      # Out5
      x = self.layer5(out2)

      # Layer123 and layer4 Fuse
      fuse_out1234_downsample = self.layer4_downsample(fuse_out1234)
      fuse_out12345 = self.fuse_mode12345(x, fuse_out1234_downsample)

      emb = self.pool(fuse_out12345)
      emb = emb.reshape(b, ts, -1)

    else:
      emb = self.pool(fuse_out1234)
      emb = emb.reshape(b, ts, -1)

    return emb


if __name__ == "__main__":
  import time
  from ..model_utils import ModelWrapper, LinearClassifer

  # feat_dim is the parameter that correspond to the number of frequency bins
  model = ERes2Net(m_channels=16,
                   feat_dim=128,
                   num_blocks=[3, 4, 6, 3, 3],
                   add_layer=True)

  class_head = LinearClassifer(emb_size=model.emb_size,
                               n_classes=527,
                               pooling="attention",
                               activation_att="sigmoid")

  # Plugging both model together
  full_model = ModelWrapper(model=model,
                            classification_head=class_head)

  full_model = full_model.to("cuda")

  loss_fn = torch.nn.BCELoss()
  optimizer = torch.optim.AdamW(params=full_model.parameters(),
                                lr=0.001)

  # Simulating an input of a batch of 64 mel spectrogram of 98 time steps
  # and 128 frequency bins
  audio_example = torch.randn(size=(12, 10, 128, 98))
  audio_example = audio_example.to("cuda")
  target = torch.rand((12, 527)) - 0.2 > 0.5
  target = target.type(torch.float)
  target = target.to("cuda")

  n_steps = 20

  start = time.time()
  for i in range(n_steps):
    optimizer.zero_grad()

    pred, _ = full_model(audio_example)
    loss = loss_fn(pred, target)

    loss.backward()
    optimizer.step()

  stop = time.time()
  print((stop - start) / n_steps)
