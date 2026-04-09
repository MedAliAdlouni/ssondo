"""MobileNetV3 model for quantization-aware training (QAT) with quantization
support.

Code adapted from: https://github.com/fschmid56/EfficientAT/tree/main/models/mn
Which is an adapted version of MobileNetV3 https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
"""
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.ops.misc import Conv2dNormActivation

from .utils import cnn_out_size, get_layers_to_fuse
from .block_types import QuantizableInvertedResidual
from training_ssondo.utils.student_models.MobileNetV3.model import InvertedResidualConfig, _mobilenet_v3_conf
from training_ssondo.utils.student_models.utils import count_parameters


class QuantizableMN(nn.Module):
  """
  MobileNet V3 main class with quantization support.

  Parameters
  ----------
  inverted_residual_setting : List[InvertedResidualConfig]
    Network structure.
  last_channel : int
    The number of channels on the penultimate layer.
  num_classes : int, optional
    Number of classes, by default 1000.
  block : Optional[Callable[..., nn.Module]], optional
    Module specifying inverted residual building block for models
    (default is None).
  norm_layer : Optional[Callable[..., nn.Module]], optional
    Module specifying the normalization layer to use (default is None).
  dropout : float, optional
    The droupout probability (default is 0.2).
  in_conv_kernel : int, optional
    Size of kernel for first convolution (default is 3).
  in_conv_stride : int, optional
    Size of stride for first convolution (default is 2).
  in_channels : int, optional
    Number of input channels (default is 1).
  """

  def __init__(
          self,
          inverted_residual_setting: List[InvertedResidualConfig],
          last_channel: int,
          num_classes: int = 1000,
          block: Optional[Callable[..., nn.Module]] = None,
          norm_layer: Optional[Callable[..., nn.Module]] = None,
          dropout: float = 0.2,
          in_conv_kernel: int = 3,
          in_conv_stride: int = 2,
          in_channels: int = 1,
          **kwargs,
  ) -> None:
    super(QuantizableMN, self).__init__()

    self.quant = QuantStub()
    self.dequant = DeQuantStub()

    if not inverted_residual_setting:
      raise ValueError("The inverted_residual_setting should not be empty")
    elif not (
        isinstance(inverted_residual_setting, Sequence)
        and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
    ):
      raise TypeError(
          "The inverted_residual_setting should be List[InvertedResidualConfig]")

    if block is None:
      block = QuantizableInvertedResidual

    # relu_only is a flag to use only ReLU instead of Hardswish compared to the
    # original model
    relu_only = kwargs.get("relu_only", False)

    depthwise_norm_layer = norm_layer = norm_layer if norm_layer is not None else partial(
        nn.BatchNorm2d, eps=0.001, momentum=0.01)

    layers = []

    kernel_sizes = [in_conv_kernel]
    strides = [in_conv_stride]

    # building first layer
    firstconv_output_channels = inverted_residual_setting[0].input_channels
    layers.append(
        Conv2dNormActivation(
            in_channels,
            firstconv_output_channels,
            kernel_size=in_conv_kernel,
            stride=in_conv_stride,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU if relu_only else nn.Hardswish,
        )
    )

    # get squeeze excitation config
    se_cnf = kwargs.get("se_conf", None)

    # building inverted residual blocks
    # - keep track of size of frequency and time dimensions for possible
    # application of Squeeze-and-Excitation on the frequency/time dimension
    # - applying Squeeze-and-Excitation on the time dimension is not
    # recommended as this constrains the network to a particular length of the
    # audio clip, whereas Squeeze-and-Excitation on the frequency bands is
    # fine, as the number of frequency bands is usually not changing
    f_dim, t_dim = kwargs.get("input_dims", (128, 1000))

    # take into account first conv layer
    f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
    t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
    for cnf in inverted_residual_setting:
      f_dim = cnf.out_size(f_dim)
      t_dim = cnf.out_size(t_dim)
      cnf.f_dim, cnf.t_dim = f_dim, t_dim  # update dimensions in block config
      layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))
      kernel_sizes.append(cnf.kernel)
      strides.append(cnf.stride)

    # building last several layers
    lastconv_input_channels = inverted_residual_setting[-1].out_channels
    lastconv_output_channels = 6 * lastconv_input_channels
    layers.append(
        Conv2dNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU if relu_only else nn.Hardswish,
        )
    )

    self.features = nn.Sequential(*layers)
    self.avgpool = nn.AdaptiveAvgPool2d(1)

    self.classification_head = kwargs.get("classification_head", False)
    if self.classification_head:
      self.classifier = nn.Sequential(
          nn.Linear(lastconv_output_channels, last_channel),
          nn.Hardswish(inplace=True),
          nn.Dropout(p=dropout, inplace=True),
          nn.Linear(last_channel, num_classes),
      )
    elif self.classification_head and relu_only:
      self.classifier = nn.Sequential(
          nn.Linear(lastconv_output_channels, last_channel),
          nn.ReLU(inplace=True),
          nn.Dropout(p=dropout, inplace=True),
          nn.Linear(last_channel, num_classes),
      )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
          nn.init.zeros_(m.bias)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    self.emb_size = lastconv_output_channels
    self.last_channel = last_channel

    self.n_parameters = count_parameters(self)
    print("Number of trainable parameters - QuantizableMobileNetV3: ", self.n_parameters)  # nopep8
    print("Size of the embedding: ", self.emb_size)

  def forward(self, x):
    """
    Forward pass of the model.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape [batch, time_steps, frequency_frames, time_frames].
      If the input tensor has fewer than 4 dimensions, it will be unsqueezed to
      add a dimension.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
      If `self.classification_head` is True, returns a tuple containing:
      - The output tensor after passing through the classifier.
      - The embedding tensor of shape [batch, time_steps, -1].
      If `self.classification_head` is False, returns only the embedding tensor.
    """

    if x.ndim < 4:
      x = x.unsqueeze(1)

    # Reshaping x into [batch * time_steps, 1, frequency_frames, time_frames]
    b, ts, f, tf = x.shape
    x = x.reshape(b * ts, 1, f, tf)

    x = self.quant(x)

    for _, layer in enumerate(self.features):
      x = layer(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    emb = x.reshape(b, ts, -1)
    emb = self.dequant(emb)

    if self.classification_head:
      x = self.classifier(emb)
      x = self.dequant(x)
      return x, emb

    else:
      return emb

  def fuse_model(self, is_qat: bool = False):
    """
    Fuse the layers of the model for quantization.

    This method fuses the layers of the model to prepare it for quantization.
    If `is_qat` is set to True, it uses Quantization Aware Training (QAT)
    specific fusion method; otherwise, it uses the standard fusion method.

    Parameters
    ----------
    is_qat : bool, optional
      If True, use Quantization Aware Training (QAT) specific fusion method.
      Default is False.
    """
    layers_to_fuse = get_layers_to_fuse(self)

    method = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
    method(self, layers_to_fuse, inplace=True)


def _quantizable_mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained_path: Optional[str],
    **kwargs: Any,
) -> QuantizableMN:

  model = QuantizableMN(inverted_residual_setting, last_channel, **kwargs)

  if pretrained_path is not None:
    checkpoint = torch.load(pretrained_path)

    # just load the state dict of the model and ignore the classification head
    state_dict = {
        k.replace("student_model.model.", ""): v for k, v in checkpoint["state_dict"].items()  # nopep8
        if not k.startswith("student_model.classification_head")
    }
    model.load_state_dict(state_dict)

  return model


def quantizable_mobilenet_v3(
        pretrained_path: Optional[str],
        **kwargs: Any,
) -> QuantizableMN:
  """
  Constructs a MobileNetV3 architecture from
  "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>".
  """
  inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
  return _quantizable_mobilenet_v3(inverted_residual_setting,
                                   last_channel,
                                   pretrained_path,
                                   **kwargs)


def get_quantizable_model(
        num_classes: int = 527,
        pretrained_path: Optional[str] = None,
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
        relu_only: bool = False,
        classification_head: bool = False,
        input_dim_f: int = 128,
        input_dim_t: int = 1000,
        se_dims: str = "c",
        se_agg: str = "max",
        se_r: int = 4,
) -> QuantizableMN:
  """
  Instantiate a MobileNetv3 model with specified configurations.

  Parameters
  ----------
  num_classes : int, optional
    Specifies number of classes to predict (default is 527).
  pretrained_name : str, optional
    Specifies name of pre-trained model to load (default is None).
  width_mult : float, optional
    Scales width of network (default is 1.0).
  reduced_tail : bool, optional
    Scales down network tail (default is False).
  dilated : bool, optional
    Applies dilated convolution to network tail (default is False).
  strides : tuple of int, optional
    Strides that are set to '2' in original implementation; might be changed
    to modify the size of receptive field and the downsampling factor in time
    and frequency dimension (default is (2, 2, 2, 2)).
  relu_only : bool, optional
    Decides whether to use only ReLU instead of Hardswish (default is False).
  classification_head : bool, optional
    Decides whether a classification head is plugged to the end of the model
    (default is False).
  input_dim_f : int, optional
    Number of frequency bands (default is 128).
  input_dim_t : int, optional
    Number of time frames (default is 1000).
  se_dims : str, optional
    Contains letters corresponding to dimensions:
      'c' - channel | 'f' - frequency | 't' - time (default is "c").
  se_agg : str, optional
    Operation to fuse output of concurrent se layers (default is "max").
  se_r : int, optional
    Squeeze excitation bottleneck size (default is 4).

  Returns
  -------
  mn : MobileNetv3
    An instance of the MobileNetv3 model with the specified configurations.
  """

  dim_map = {"c": 1, "f": 2, "t": 3}
  assert len(se_dims) <= 3 and all([s in dim_map.keys() for s in se_dims]) or se_dims == "none"  # nopep8

  input_dims = (input_dim_f, input_dim_t)

  if se_dims == "none":
    se_dims = None  # type: ignore
  else:
    se_dims = [dim_map[s] for s in se_dims]  # type: ignore
  se_conf = dict(se_dims=se_dims, se_agg=se_agg, se_r=se_r)

  mn = quantizable_mobilenet_v3(
      num_classes=num_classes,
      pretrained_path=pretrained_path,
      width_mult=width_mult,
      reduced_tail=reduced_tail,
      dilated=dilated,
      strides=strides,
      relu_only=relu_only,
      classification_head=classification_head,
      input_dims=input_dims,
      se_conf=se_conf,
  )

  return mn
