from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.misc import ConvNormActivation
import torch

from .dy_block import DynamicInvertedResidualConfig, DY_Block, DynamicConv, DyReLUB
from .block_types import InvertedResidualConfig, InvertedResidual


class DyMN(nn.Module):
  def __init__(
          self,
          inverted_residual_setting: List[DynamicInvertedResidualConfig],
          last_channel: int,
          num_classes: int = 527,
          head_type: str = "mlp",
          block: Optional[Callable[..., nn.Module]] = None,
          norm_layer: Optional[Callable[..., nn.Module]] = None,
          dropout: float = 0.2,
          in_conv_kernel: int = 3,
          in_conv_stride: int = 2,
          in_channels: int = 1,
          context_ratio: int = 4,
          max_context_size: int = 128,
          min_context_size: int = 32,
          dyrelu_k=2,
          dyconv_k=4,
          no_dyrelu: bool = False,
          no_dyconv: bool = False,
          no_ca: bool = False,
          temp_schedule: tuple = (30, 1, 1, 0.05),
          **kwargs: Any,
  ) -> None:
    super(DyMN, self).__init__()

    if not inverted_residual_setting:
      raise ValueError("The inverted_residual_setting should not be empty")
    elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, DynamicInvertedResidualConfig) for s in inverted_residual_setting])
    ):
      raise TypeError(
          "The inverted_residual_setting should be List[DynamicInvertedResidualConfig]")

    if block is None:
      block = DY_Block

    norm_layer = \
        norm_layer if norm_layer is not None else partial(
            nn.BatchNorm2d, eps=0.001, momentum=0.01)

    self.layers = nn.ModuleList()

    # building first layer
    firstconv_output_channels = inverted_residual_setting[0].input_channels
    self.in_c = ConvNormActivation(
        in_channels,
        firstconv_output_channels,
        kernel_size=in_conv_kernel,
        stride=in_conv_stride,
        norm_layer=norm_layer,
        activation_layer=nn.Hardswish,
    )

    for cnf in inverted_residual_setting:
      if cnf.use_dy_block:
        b = block(cnf,
                  context_ratio=context_ratio,
                  max_context_size=max_context_size,
                  min_context_size=min_context_size,
                  dyrelu_k=dyrelu_k,
                  dyconv_k=dyconv_k,
                  no_dyrelu=no_dyrelu,
                  no_dyconv=no_dyconv,
                  no_ca=no_ca,
                  temp_schedule=temp_schedule
                  )
      else:
        b = InvertedResidual(cnf, None, norm_layer, partial(
            nn.BatchNorm2d, eps=0.001, momentum=0.01))

      self.layers.append(b)

    # building last several layers
    lastconv_input_channels = inverted_residual_setting[-1].out_channels
    lastconv_output_channels = 6 * lastconv_input_channels
    self.out_c = ConvNormActivation(
        lastconv_input_channels,
        lastconv_output_channels,
        kernel_size=1,
        norm_layer=norm_layer,
        activation_layer=nn.Hardswish,
    )

    self.head_type = head_type

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    self.emb_size = lastconv_output_channels
    self.last_channel = last_channel

  def _feature_forward(self, x: Tensor) -> (Tensor, Tensor):
    x = self.in_c(x)
    g = None
    for layer in self.layers:
      x = layer(x)
    x = self.out_c(x)
    return x

  def _forward_impl(self, x: Tensor) -> (Tensor, Tensor):
    x = self._feature_forward(x)
    embed = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
    if embed.ndim < 3:
      embed = embed.unsqueeze(dim=1)
    return embed

  def forward(self, x: Tensor) -> (Tensor, Tensor):
    return self._forward_impl(x)

  def update_params(self, epoch):
    for module in self.modules():
      if isinstance(module, DynamicConv):
        module.update_params(epoch)


def _dymn_conf(
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int] = (2, 2, 2, 2),
        use_dy_blocks: str = "all",
        **kwargs: Any
):
  reduce_divider = 2 if reduced_tail else 1
  dilation = 2 if dilated else 1

  bneck_conf = partial(DynamicInvertedResidualConfig, width_mult=width_mult)
  adjust_channels = partial(
      DynamicInvertedResidualConfig.adjust_channels, width_mult=width_mult)

  activations = ["RE", "RE", "RE", "RE", "RE", "RE",
                 "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS"]

  if use_dy_blocks == "all":
    # per default the dynamic blocks replace all conventional IR blocks
    use_dy_block = [True] * 15
  elif use_dy_blocks == "replace_se":
    use_dy_block = [False, False, False, True, True, True,
                    False, False, False, False, True, True, True, True, True]
  else:
    raise NotImplementedError(
        f"Config use_dy_blocks={use_dy_blocks} not implemented.")

  inverted_residual_setting = [
      bneck_conf(16, 3, 16, 16, use_dy_block[0], activations[0], 1, 1),
      bneck_conf(16, 3, 64, 24, use_dy_block[1],
                 activations[1], strides[0], 1),  # C1
      bneck_conf(24, 3, 72, 24, use_dy_block[2], activations[2], 1, 1),
      bneck_conf(24, 5, 72, 40, use_dy_block[3],
                 activations[3], strides[1], 1),  # C2
      bneck_conf(40, 5, 120, 40, use_dy_block[4], activations[4], 1, 1),
      bneck_conf(40, 5, 120, 40, use_dy_block[5], activations[5], 1, 1),
      bneck_conf(40, 3, 240, 80,
                 use_dy_block[6], activations[6], strides[2], 1),  # C3
      bneck_conf(80, 3, 200, 80, use_dy_block[7], activations[7], 1, 1),
      bneck_conf(80, 3, 184, 80, use_dy_block[8], activations[8], 1, 1),
      bneck_conf(80, 3, 184, 80, use_dy_block[9], activations[9], 1, 1),
      bneck_conf(80, 3, 480, 112, use_dy_block[10], activations[10], 1, 1),
      bneck_conf(112, 3, 672, 112, use_dy_block[11], activations[11], 1, 1),
      bneck_conf(112, 5, 672, 160 // reduce_divider,
                 use_dy_block[12], activations[12], strides[3], dilation),  # C4
      bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[13],
                 activations[13], 1, dilation),
      bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[14],
                 activations[14], 1, dilation),
  ]
  last_channel = adjust_channels(1280 // reduce_divider)

  return inverted_residual_setting, last_channel


def _dymn(
        inverted_residual_setting: List[DynamicInvertedResidualConfig],
        last_channel: int,
        **kwargs: Any,
):
  model = DyMN(inverted_residual_setting, last_channel, **kwargs)
  return model


def dymn(pretrained_name: str = None, **kwargs: Any):
  inverted_residual_setting, last_channel = _dymn_conf(**kwargs)
  return _dymn(inverted_residual_setting, last_channel, **kwargs)


def get_model(num_classes: int = 527,
              pretrained_name: str = None,
              width_mult: float = 1.0,
              strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
              # Context
              context_ratio: int = 4,
              max_context_size: int = 128,
              min_context_size: int = 32,
              # Dy-ReLU
              dyrelu_k: int = 2,
              no_dyrelu: bool = False,
              # Dy-Conv
              dyconv_k: int = 4,
              no_dyconv: bool = False,
              T_max: float = 30.0,
              T0_slope: float = 1.0,
              T1_slope: float = 0.02,
              T_min: float = 1,
              pretrain_final_temp: float = 1.0,
              # Coordinate Attention
              no_ca: bool = False,
              use_dy_blocks="all"):
  """
  Arguments to modify the instantiation of a DyMN.

  Args:
      num_classes (int): Specifies number of classes to predict
      pretrained_name (str): Ignored at inference time
      width_mult (float): Scales width of network
      strides (Tuple): Strides that are set to '2' in original implementation
      context_ratio (int): fraction of expanded channel representation used as context size
      max_context_size (int): maximum size of context
      min_context_size (int): minimum size of context
      dyrelu_k (int): number of linear mappings
      no_dyrelu (bool): not use Dy-ReLU
      dyconv_k (int): number of kernels for dynamic convolution
      no_dyconv (bool): not use Dy-Conv
      T_max, T0_slope, T1_slope, T_min (float): hyperparameters for temperature schedule
      pretrain_final_temp (float): final Dy-Conv temperature used at inference
      no_ca (bool): not use Coordinate Attention
      use_dy_blocks (str): use dynamic block at all positions per default
  """

  block = DY_Block
  # At inference, use the final temperature from training
  T_max = pretrain_final_temp

  temp_schedule = (T_max, T_min, T0_slope, T1_slope)

  m = dymn(num_classes=num_classes,
           pretrained_name=pretrained_name,
           block=block,
           width_mult=width_mult,
           strides=strides,
           context_ratio=context_ratio,
           max_context_size=max_context_size,
           min_context_size=min_context_size,
           dyrelu_k=dyrelu_k,
           dyconv_k=dyconv_k,
           no_dyrelu=no_dyrelu,
           no_dyconv=no_dyconv,
           no_ca=no_ca,
           temp_schedule=temp_schedule,
           use_dy_blocks=use_dy_blocks
           )
  return m
