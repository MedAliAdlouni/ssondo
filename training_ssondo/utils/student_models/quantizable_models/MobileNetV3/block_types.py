from typing import Dict, Callable, List

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation

from training_ssondo.utils.student_models.MobileNetV3.utils import make_divisible
from training_ssondo.utils.student_models.MobileNetV3.model import (
    InvertedResidualConfig,
)


class QuantizableConcurrentSEBlock(nn.Module):
    """
    A concurrent Squeeze-and-Excitation (SE) block that supports quantization.

    This block applies multiple SE layers concurrently and aggregates their
    outputs using a specified aggregation operation (max, avg, add, min).

    Parameters
    ----------
    c_dim : int
      The channel dimension.
    f_dim : int
      The frequency dimension.
    t_dim : int
      The time dimension.
    se_cnf : Dict
      Configuration dictionary for SE layers. It should contain:
      - 'se_dims': List of dimensions to apply SE layers.
      - 'se_r': Reduction ratio for SE layers.
      - 'se_agg': Aggregation operation ('max', 'avg', 'add', 'min').

    Attributes
    ----------
    conc_se_layers : nn.ModuleList
      A list of SE layers to be applied concurrently.
    agg : bool
      A flag indicating whether to aggregate the outputs of multiple SE layers.
    agg_op : Callable
      The aggregation operation to combine the outputs of multiple SE layers.

    Methods
    -------
    forward(input: Tensor) -> Tensor
      Applies the concurrent SE layers and aggregates their outputs.
    """

    def __init__(self, c_dim: int, f_dim: int, t_dim: int, se_cnf: Dict) -> None:
        super().__init__()

        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()

        for d in se_cnf["se_dims"]:
            input_dim = dims[d - 1]
            squeeze_dim = make_divisible(input_dim // se_cnf["se_r"], 8)
            self.conc_se_layers.append(
                QuantizableSqueezeExcitation(input_dim, squeeze_dim, d)
            )  # nopep8

        if len(se_cnf["se_dims"]) == 1:
            self.agg = False
        else:
            self.agg = True

            if se_cnf["se_agg"] == "max":
                self.agg_op = lambda x: torch.max(x, dim=0)[0]
            elif se_cnf["se_agg"] == "avg":
                self.agg_op = lambda x: torch.mean(x, dim=0)
            elif se_cnf["se_agg"] == "add":
                self.agg_op = lambda x: torch.sum(x, dim=0)
            elif se_cnf["se_agg"] == "min":
                self.agg_op = lambda x: torch.min(x, dim=0)[0]
            else:
                raise NotImplementedError(
                    f"SE aggregation operation '{self.agg_op}' not implemented"
                )

    def forward(self, input: Tensor) -> Tensor:
        if not self.agg:
            # Applies the SE layer.
            return self.conc_se_layers[-1](input)

        else:
            # Applies the concurrent SE layers and aggregates their outputs.
            # Note: Aggregation operations are not supported by onnx export.
            se_outs = []
            for se_layer in self.conc_se_layers:
                se_outs.append(se_layer(input))
            out = self.agg_op(torch.stack(se_outs, dim=0))
            return out


class QuantizableSqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from
    https://arxiv.org/abs/1709.01507.
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta``
    and ``sigma`` in eq. 3.

    Parameters
    ----------
    input_dim : int
      Input dimension.
    squeeze_dim : int
      Size of Bottleneck.
    se_dim : int
      Dimension for squeeze-and-excitation. Must be one of [1, 2, 3].
    activation : Callable[..., torch.nn.Module], optional
      Activation applied to bottleneck (default is nn.ReLU).
    scale_activation : Callable[..., torch.nn.Module], optional
      Activation applied to the output (default is nn.Sigmoid).
    """

    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)

        assert se_dim in [1, 2, 3]
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)

        self.activation = activation()
        self.scale_activation = scale_activation()
        self.skip_mul = torch.ao.nn.quantized.FloatFunctional()

    def _scale(self, input: Tensor) -> Tensor:
        scale = torch.mean(input, self.se_dim, keepdim=True)
        shape = scale.size()

        # Squeeze the dimensions that are not used for the SE operation.
        # Note: squeeze() does not work for onnx export.
        if self.se_dim == [2, 3]:
            scale = self.fc1(scale[:, :, 0, 0])
        elif self.se_dim == [1, 3]:
            scale = self.fc1(scale[:, 0, :, 0])
        else:
            scale = self.fc1(scale[:, 0, 0, :])

        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale).view(shape)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return self.skip_mul.mul(scale, input)


class QuantizableInvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: Dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf["se_dims"] is not None:
            layers.append(
                QuantizableConcurrentSEBlock(
                    cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf
                )
            )

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.skip_add.add(input, result)
        return result
