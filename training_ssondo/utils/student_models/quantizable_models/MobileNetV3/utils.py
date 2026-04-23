import math
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch import Tensor


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(
    x: Tensor,
    dim: int,
    mode: str = "pool",
    pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
    combine_dim: int = None,
):
    """
    Collapses dimension of multi-dimensional tensor by pooling or combining dimensions
    :param x: input Tensor
    :param dim: dimension to collapse
    :param mode: 'pool' or 'combine'
    :param pool_fn: function to be applied in case of pooling
    :param combine_dim: dimension to join 'dim' to
    :return: collapsed tensor
    """
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)


class CollapseDim(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str = "pool",
        pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
        combine_dim: int = None,
    ):
        super(CollapseDim, self).__init__()
        self.dim = dim
        self.mode = mode
        self.pool_fn = pool_fn
        self.combine_dim = combine_dim

    def forward(self, x):
        return collapse_dim(
            x,
            dim=self.dim,
            mode=self.mode,
            pool_fn=self.pool_fn,
            combine_dim=self.combine_dim,
        )


def get_layers_to_fuse(model):
    """
    Identify and group layers in a model for fusion.

    This function iterates through the layers of a given model to identify groups
    of layers that can be fused together. It specifically targets sequences of
    Conv2d, BatchNorm2d, and ReLU layers, or just Conv2d and BatchNorm2d layers,
    that can be fused to optimize the model for inference.

    Parameters
    ----------
    model : nn.Module
      The neural network model containing the layers to be analyzed.

    Returns
    -------
    layers_to_fuse : list of list of str
      A list where each element is a list of layer names that can be fused
      together.
    """

    layers_to_fuse = []
    layer_names = []

    prev_type = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and prev_type in [None, "relu"]:
            # If the current layer is Conv2d and the previous layer was None or ReLU
            if layer_names:
                # If there are layers in layer_names, add them to layers_to_fuse
                layers_to_fuse.append(layer_names)

            # Start a new group with the current Conv2d layer and update prev_type
            layer_names = [name]
            prev_type = "conv"

        elif isinstance(module, nn.BatchNorm2d) and prev_type == "conv":
            # If the current layer is BatchNorm2d and the previous layer was Conv2d
            # Add the current BatchNorm2d layer to the current group and update
            # prev_type
            layer_names.append(name)
            prev_type = "bn"

        elif isinstance(module, nn.ReLU) and prev_type == "bn":
            # If the current layer is ReLU and the previous layer was BatchNorm2d
            # Add the current ReLU layer to the current group, add the group to
            # layers_to_fuse and update prev_type
            layer_names.append(name)
            layers_to_fuse.append(layer_names)
            layer_names = []
            prev_type = "relu"

        elif prev_type == "bn":
            # If the previous layer was BatchNorm2d but the current layer is not ReLU
            # Add the current group to layers_to_fuse, reset layer_names and reset
            # prev_type
            layers_to_fuse.append(layer_names)
            layer_names = []
            prev_type = None

    if layer_names:
        # If there are any remaining layers in layer_names, add them to
        # layers_to_fuse
        layers_to_fuse.append(layer_names)

    return layers_to_fuse
