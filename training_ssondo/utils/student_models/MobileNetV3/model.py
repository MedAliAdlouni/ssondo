"""
Code adapted from https://github.com/fschmid56/EfficientAT/tree/main/models/mn
Which is an adapted version of MobileNetV3 https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
"""

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import os
from pathlib import Path
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

from .utils import cnn_out_size
from .block_types import InvertedResidualConfig, InvertedResidual

from training_ssondo.utils.student_models.utils import count_parameters


class MN(nn.Module):
    """
    MobileNet V3 main class.

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
        super(MN, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(
                [
                    isinstance(s, InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = InvertedResidual

        # relu_only is a flag to use only ReLU instead of Hardswish compared to the
        # original model
        relu_only = kwargs.get("relu_only", False)

        depthwise_norm_layer = norm_layer = (
            norm_layer
            if norm_layer is not None
            else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        )

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
        print("Number of trainable parameters - MobileNetV3: ", self.n_parameters)
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

        for _, layer in enumerate(self.features):
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        emb = x.reshape(b, ts, -1)

        if self.classification_head:
            x = self.classifier(emb)

        if self.classification_head:
            return x, emb
        else:
            return emb


def _mobilenet_v3_conf(
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
    relu_only: bool = False,
    **kwargs,
):

    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    # InvertedResidualConfig:
    # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation # nopep8
    activation = "RE" if relu_only else "HS"
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, activation, strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, False, activation, 1, 1),
        bneck_conf(80, 3, 184, 80, False, activation, 1, 1),
        bneck_conf(80, 3, 184, 80, False, activation, 1, 1),
        bneck_conf(80, 3, 480, 112, True, activation, 1, 1),
        bneck_conf(112, 3, 672, 112, True, activation, 1, 1),
        bneck_conf(
            112, 5, 672, 160 // reduce_divider, True, activation, strides[3], dilation
        ),  # C4 # nopep8
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            activation,
            1,
            dilation,
        ),  # nopep8
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            activation,
            1,
            dilation,
        ),  # nopep8
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained_name: Optional[str],
    **kwargs: Any,
):
    model = MN(inverted_residual_setting, last_channel, **kwargs)

    if pretrained_name is not None:
        # ImageNet pre-trained model from https://github.com/fschmid56/EfficientAT
        # NOTE: for easy loading, Schmid et al. provides the adapted state dict
        # ("mn10_im.pt") ready for AudioSet training (1 input channel, 527 output
        # classes)
        # NOTE: the classifier is just a random initialization, feature extractor
        # (conv layers) is pre-trained

        print(f"Loading the weights of the pre-trained model: {pretrained_name}")

        pretrained_path = os.path.join(
            _PROJECT_ROOT,
            "models",
            "students",
            "MobileNetV3",
            "pretrained_models",
            pretrained_name,
        )
        state_dict = torch.load(pretrained_path)

        if kwargs["classification_head"]:
            print("Loading pre-trained weights in a non-strict manner.")
            model.load_state_dict(state_dict, strict=False)

        else:
            # Drop classifier's weights
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("classifier")
            }

            # Load model weights
            model.load_state_dict(state_dict)

    return model


def mobilenet_v3(
    pretrained_name: Optional[str],
    **kwargs: Any,
) -> MN:
    """
    Constructs a MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>".
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return _mobilenet_v3(
        inverted_residual_setting, last_channel, pretrained_name, **kwargs
    )


def get_model(
    num_classes: int = 527,
    pretrained_name: Optional[str] = None,
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
):
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
    assert (
        len(se_dims) <= 3
        and all([s in dim_map.keys() for s in se_dims])
        or se_dims == "none"
    )  # nopep8

    input_dims = (input_dim_f, input_dim_t)

    if se_dims == "none":
        se_dims = None  # type: ignore
    else:
        se_dims = [dim_map[s] for s in se_dims]  # type: ignore
    se_conf = dict(se_dims=se_dims, se_agg=se_agg, se_r=se_r)

    mn = mobilenet_v3(
        num_classes=num_classes,
        pretrained_name=pretrained_name,
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
