"""Useful pooling layers for neural networks."""

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    MeanPooling layer.
    This layer computes the mean of the input tensor along a specified dimension.

    Parameters
    ----------
    dim : int, optional
      The dimension along which to compute the mean (default is 1).

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
      Forward pass of the MeanPooling layer.
    """

    def __init__(self, dim: int = 1) -> None:
        super(MeanPooling, self).__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the mean pooling layer.

        Parameters
        ----------
        x : torch.Tensor
          The input tensor.

        Returns
        -------
        torch.Tensor
          The tensor with the mean computed along the specified dimension.
        """

        return x.mean(dim=self.dim)


class AttentionPooling(nn.Module):
    """
    AttentionPooling layer.
    This layer applies an attention-based pooling over the input tensor.

    Parameters
    ----------
    input_size : int
      The size of the input features.
    activation : str, optional
      The activation function to use
      ('softmax' or 'sigmoid', default is 'softmax').

    Methods
    -------
    forward(x)
      Forward pass of the AttentionPooling layer.

    Raises
    ------
    ValueError
      If the activation function is not 'softmax' or 'sigmoid'.
    """

    def __init__(self, input_size: int, activation: str = "softmax") -> None:
        super(AttentionPooling, self).__init__()

        self.linear = nn.Linear(in_features=input_size, out_features=input_size)

        if activation == "sigmoid":
            self.activation_layer = nn.Sigmoid()
        elif activation == "softmax":
            self.activation_layer = nn.Softmax(dim=-1)
        else:
            raise ValueError(
                "Activation function must be either 'softmax' or 'sigmoid'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AttentionPooling layer.

        Parameters
        ----------
        x : torch.Tensor
          Input tensor of shape (batch_size, n, input_size), where `n` is the
          number of elements to pool over.

        Returns
        -------
        torch.Tensor
          Output tensor of shape (batch_size, input_size) after applying the
          attention-based pooling.
        """

        att = self.linear(x)
        att = self.activation_layer(att)
        att = torch.clamp(att, min=1e-7, max=1)

        return (x * att).sum(dim=1) / att.sum(dim=1)


class WeightedPooling(nn.Module):
    """
    WeightedPooling layer.
    This layer applies a weighted pooling over the input tensor.

    Parameters
    ----------
    n_weights : int
      The number of weights to be used for pooling.

    Attributes
    ----------
    weights : torch.nn.Parameter
      The learnable weights for the pooling layer.
    softmax : torch.nn.Softmax
      The softmax function to normalize the weights.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
      Forward pass of the weighted pooling layer.
    """

    def __init__(self, n_weights: int) -> None:
        super(WeightedPooling, self).__init__()

        self.weights = nn.Parameter(torch.ones(n_weights) / n_weights)
        self.norm_weights = None
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the weighted pooling layer.

        Parameters
        ----------
        x : torch.Tensor
          The input tensor of shape (batch_size, n, input_size), where `n` is the
          number of elements to pool over.

        Returns
        -------
        torch.Tensor
          Output tensor of shape (batch_size, input_size) after applying the
          weighted pooling.
        """

        norm_weights = self.softmax(self.weights)
        self.norm_weights = norm_weights

        x = torch.einsum("i,bim->bm", norm_weights, x)

        return x
