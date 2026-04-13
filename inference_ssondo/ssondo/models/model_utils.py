"""Code for different classification models used to obtain class scores from a
sequence of embeddings.

It contains various classifier including Linear, MLP, and RNN-based classifiers.
Each projects a sequence of embeddings into class scores using different
architectures and pooling mechanisms.
"""
import torch
import torch.nn as nn
from ssondo.models.utils import count_parameters

from ssondo.models.pooling_layers import AttentionPooling, MeanPooling


class LinearClassifer(nn.Module):
  """
  A linear classifier model with configurable pooling and activation functions.
  It projects a sequence of embeddings into a class scores.

  Parameters
  ----------
  emb_size : int
    The size of the input embeddings.
  n_classes : int
    The number of output classes.
  pooling : str or None, optional
    The type of pooling to use
    ('attention', 'mean' or None, default is 'attention').
  activation_att : str, optional
    The activation function to use for the attention pooling mechanism, no
    effect if pooling is 'mean' ('sigmoid' or 'softmax', default is 'sigmoid').
  last_activation : str, optional
    The activation function to use for the output
    ('sigmoid' or '', default is 'sigmoid').
  """

  def __init__(self,
               emb_size,
               n_classes,
               pooling="attention",
               activation_att="sigmoid",
               last_activation="sigmoid") -> None:
    super(LinearClassifer, self).__init__()

    self.linear = nn.Linear(
        in_features=emb_size,
        out_features=n_classes
    )

    # Defines the pooling operation
    if pooling == "attention":
      self.pooling = AttentionPooling(input_size=n_classes,
                                      activation=activation_att)
    elif pooling == "mean":
      self.pooling = MeanPooling(dim=1)

    elif pooling is None:
      self.pooling = None

    else:
      raise ValueError

    # Defines the activation used for output
    if last_activation == "sigmoid":
      self.last_activation = nn.Sigmoid()
    elif last_activation == "softmax":
      self.last_activation = nn.Softmax(dim=-1)
    elif last_activation == "":
      self.last_activation = nn.Identity()
    else:
      raise ValueError

    self.n_parameters = count_parameters(self)

  def forward(self, x):
    out = self.linear(x)
    out = self.pooling(out) if self.pooling is not None else out
    out = self.last_activation(out)

    return out


class MLPClassifer(nn.Module):
  """
  Multi-Layer Perceptron (MLP) Classifier with configurable pooling and
  activation. It projects a sequence of embeddings into class scores.

  Parameters
  ----------
  emb_size : int
    The size of the input embeddings.
  n_classes : int
    The number of output classes.
  hidden_features_size : int, optional
    The number of hidden features in the MLP (default is 768).
  dropout : float, optional
    The dropout rate (default is 0.2).
  pooling : str, or None, optional
    The type of pooling to use
    ('attention', 'mean' or None, default is 'attention').
  activation_att : str, optional
    The activation function to use for the attention pooling mechanism, no
    effect if pooling is 'mean' ('sigmoid' or 'softmax', default is 'sigmoid').
  last_activation : str, optional
    The activation function to use for the output
    ('sigmoid' or '', default is 'sigmoid').
  """

  def __init__(self,
               emb_size,
               n_classes,
               hidden_features_size=768,
               dropout=0.2,
               pooling="attention",
               activation_att="sigmoid",
               last_activation="sigmoid",) -> None:
    super(MLPClassifer, self).__init__()

    self.mlp = nn.Sequential(
        nn.Linear(in_features=emb_size, out_features=hidden_features_size),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(hidden_features_size, n_classes)
    )

    # Defines the pooling operation
    if pooling == "attention":
      self.pooling = AttentionPooling(input_size=n_classes,
                                      activation=activation_att)
    elif pooling == "mean":
      self.pooling = MeanPooling(dim=1)

    elif pooling is None:
      self.pooling = None

    else:
      raise ValueError

    # Defines the activation used for output
    if last_activation == "sigmoid":
      self.last_activation = nn.Sigmoid()
    elif last_activation == "softmax":
      self.last_activation = nn.Softmax(dim=-1)
    elif last_activation == "":
      self.last_activation = nn.Identity()
    else:
      raise ValueError

    self.n_parameters = count_parameters(self)

  def forward(self, x):
    out = self.mlp(x)
    out = self.pooling(out) if self.pooling is not None else out
    out = self.last_activation(out)

    return out


class RNNClassifer(nn.Module):
  """
  A Recurrent Neural Network (RNN) based classifier. It projects a sequence of
  embeddings into class scores.
  """

  def __init__(self,
               rnn_type,
               emb_size,
               hidden_size,
               n_classes,
               num_layers=1,
               batch_first=True,
               bidirectional=False,
               linear_in_activation=nn.Identity(),
               n_last_elements=1,
               last_activation="sigmoid") -> None:
    super(RNNClassifer, self).__init__()

    # Linear layer to transform input embeddings to hidden size
    self.linear_in = nn.Linear(
        in_features=emb_size,
        out_features=hidden_size,
    )
    self.linear_in_activation = linear_in_activation

    # RNN layer
    self.rnn = getattr(nn, rnn_type.upper())(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
    )

    # Linear layer to transform RNN output to number of classes
    self.linear_out = nn.Linear(
        in_features=hidden_size * 2 if bidirectional else hidden_size,
        out_features=n_classes,
    )

    # Number of last elements to consider for mean pooling
    self.n_last_elements = n_last_elements

    # Defines the activation used for output
    if last_activation == "sigmoid":
      self.last_activation = nn.Sigmoid()
    elif last_activation == "":
      self.last_activation = nn.Identity()
    else:
      raise ValueError

    self.n_parameters = count_parameters(self)

  def forward(self, x):
    out = self.linear_in(x)  # shape [bs, n_seq, hidden_size]
    out = self.linear_in_activation(out)  # shape [bs, n_seq, hidden_size]
    out, _ = self.rnn(out)  # shape [bs, n_seq, hidden_size]
    out = self.linear_out(out)  # shape [bs, n_seq, n_classes]

    if self.n_last_elements is not None:
      # Take the mean of the last n elements, shape [bs, n_classes]
      out = out[:, -self.n_last_elements:, :].mean(dim=1)

    # Applies the last activation function
    out = self.last_activation(out)  # shape [bs, n_classes]

    return out


class AttentionRNNClassifer(nn.Module):
  """
  AttentionRNNClassifer is a neural network model that combines an RNN with an
  attention mechanism for sequence classification tasks.
  """

  def __init__(self,
               rnn_type,
               emb_size,
               hidden_size,
               n_classes,
               num_layers=1,
               batch_first=True,
               bidirectional=False,
               linear_in_activation=nn.Identity(),
               n_last_elements=1,
               last_activation="sigmoid") -> None:
    super(AttentionRNNClassifer, self).__init__()

    # Linear layer to transform input embeddings to hidden size
    self.linear_in = nn.Linear(
        in_features=emb_size,
        out_features=hidden_size,
    )
    self.linear_in_activation = linear_in_activation

    # RNN layer
    self.rnn = getattr(nn, rnn_type.upper())(
        input_size=hidden_size,
        hidden_size=hidden_size // 2 if bidirectional else hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
    )

    # Attention layer to compute context
    self.attention = BahdanauAttention(hidden_size)

    # Linear layer to transform RNN output to number of classes
    self.linear_out = nn.Linear(
        in_features=hidden_size * 2,
        out_features=n_classes,
    )

    # Number of last elements to consider for mean pooling
    self.n_last_elements = n_last_elements

    # Defines the activation used for output
    if last_activation == "sigmoid":
      self.last_activation = nn.Sigmoid()
    elif last_activation == "":
      self.last_activation = nn.Identity()
    else:
      raise ValueError

    self.n_parameters = count_parameters(self)

  def forward(self, x, hidden=None):
    outputs = []

    x = self.linear_in(x)  # shape [bs, n_seq, hidden_size]
    x = self.linear_in_activation(x)  # shape [bs, n_seq, hidden_size]

    for i in range(1, x.size(1) + 1):
      out, hidden = self.rnn(x[:, :i], hidden)  # shape [bs, n, hidden_size]
      context, _ = self.attention(out, x[:, :i])

      cats = torch.cat((out[:, -1:], context), dim=2)
      cats = cats.squeeze(1)

      outputs.append(self.linear_out(cats))

    outputs = torch.stack(outputs, dim=1)  # shape [bs, n_seq, n_classes]

    if self.n_last_elements is not None:
      # Take the mean of the last n elements, shape [bs, n_classes]
      out = out[:, -self.n_last_elements:, :].mean(dim=1)

    # Applies the last activation function
    outputs = self.last_activation(outputs)  # shape [bs, n_classes]

    return outputs


class BahdanauAttention(nn.Module):
  """Bahdanau Attention mechanism."""

  def __init__(self, hidden_size):
    super(BahdanauAttention, self).__init__()
    self.wa = nn.Linear(hidden_size, hidden_size)
    self.ua = nn.Linear(hidden_size, hidden_size)
    self.va = nn.Linear(hidden_size, 1)

  def forward(self, query, keys):
    scores = self.va(torch.tanh(self.wa(query) + self.ua(keys)))
    scores = scores.squeeze(2).unsqueeze(1)

    weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.bmm(weights, keys)

    return context, weights


class ModelWrapper(nn.Module):
  """
  A wrapper class for a model and its classification head.

  This class combines a base model and a classification head into a single
  module. It first passes the input through the base model to obtain embeddings,
  and then passes these embeddings through the classification head to obtain
  classification outputs.

  Parameters
  ----------
  model : nn.Module
    The base model used to generate embeddings from the input data.
  classification_head : nn.Module
    The classification head used to generate classification outputs from the
    embeddings.

  Methods
  -------
  forward(x)
    Performs a forward pass through the model and classification head.
  """

  def __init__(self, model, classification_head) -> None:
    super(ModelWrapper, self).__init__()

    self.model = model
    self.classification_head = classification_head

  def forward(self, x):
    """
    Forward pass through the model.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape [batch_size, n_seq, input_size].

    Returns
    -------
    cla : torch.Tensor
      Output tensor of shape [batch_size, n_classes] from the classification
      head.
    emb : torch.Tensor
      Embedding tensor of shape [batch_size, n_seq, emb_size] from the model.
    """
    emb = self.model(x)
    cla = self.classification_head(emb)

    return cla, emb
