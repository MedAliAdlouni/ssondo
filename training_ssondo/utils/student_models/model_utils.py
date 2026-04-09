"""Code for different classification models used to obtain class scores from a
sequence of embeddings.

It contains various classifier including Linear, MLP, and RNN-based classifiers.
Each projects a sequence of embeddings into class scores using different
architectures and pooling mechanisms.
"""
import torch
import torch.nn as nn
from training_ssondo.utils.student_models.utils import count_parameters

from training_ssondo.utils.student_models.pooling_layers import AttentionPooling, MeanPooling


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

  Attributes
  ----------
  linear : nn.Linear
    The linear transformation layer.
  pooling : nn.Module or None
    The pooling operation layer.
  last_activation : nn.Module
    The activation function for the output layer.
  n_parameters : int
    The number of trainable parameters in the model.

  Methods
  -------
  forward(x)
    Forward pass of the model.

  Raises
  ------
  ValueError
    If an invalid pooling or last_activation option is provided.
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
    print("Number of trainable parameters - Linear Classifier: ",
          self.n_parameters)

  def forward(self, x):
    """
    Perform a forward pass through the model.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape [batch_size, n_seq, input_size] to the model.

    Returns
    -------
    out : torch.Tensor
      Output tensor of shape [batch_size, n_classes] after passing through
      the MLP classifier.
    """

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

  Attributes
  ----------
  mlp : nn.Sequential
    The MLP layers.
  pooling : nn.Module or None
    The pooling layer.
  last_activation : nn.Module
    The activation function for the output.
  n_parameters : int
    The number of trainable parameters in the model.

  Methods
  -------
  forward(x)
    Forward pass through the network.

  Raises
  ------
  ValueError
    If an invalid pooling or last_activation type is provided.

  Notes
  -----
  This reproduces the [classifier](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L190)
  used in [MobileNetV3](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py).
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
    print("Number of trainable parameters - MLP Classifier: ",
          self.n_parameters)

  def forward(self, x):
    """
    Perform a forward pass through the model.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape [batch_size, n_seq, input_size].

    Returns
    -------
    out : torch.Tensor
      Output tensor of shape [batch_size, n_classes] after passing through
      the MLP classifier.
    """

    out = self.mlp(x)
    out = self.pooling(out) if self.pooling is not None else out
    out = self.last_activation(out)

    return out


class RNNClassifer(nn.Module):
  """
  A Recurrent Neural Network (RNN) based classifier. It projects a sequence of
  embeddings into class scores.

  Parameters
  ----------
  rnn_type : str
    Type of RNN to use (e.g., 'lstm', 'gru' or 'rnn').
  emb_size : int
    The size of the input embeddings.
  hidden_size : int
    The number of features in the hidden state.
  n_classes : int
    The number of output classes.
  num_layers : int, optional
    The number of recurrent layers (default is 1).
  batch_first : bool, optional
    If True, then the input and output tensors are provided as
    (batch, seq, feature) (default is False).
  bidirectional : bool, optional
    If True, becomes a bidirectional RNN (default is False).
  linear_in_activation : nn.Module, optional
    Activation function applied after the linear_in layer
    (default is nn.Identity()).
  n_last_elements : int or None, optional
    Number of last elements from the sequence to consider for mean pooling
    (default is 1).
  last_activation : str, optional
    The activation function to use for the output
    ('sigmoid' or '', default is 'sigmoid').

  Attributes
  ----------
  linear_in : nn.Linear
    Linear layer to transform input embeddings to hidden size.
  linear_in_activation : nn.Module
    Activation function applied after the linear_in layer.
  rnn : nn.Module
    The RNN layer (LSTM, GRU or RNN).
  linear_out : nn.Linear
    Linear layer to transform RNN output to number of classes.
  n_last_elements: int or None
    Number of last elements from the sequence to consider for mean pooling.
  last_activation : nn.Module
    Activation function applied to the output.
  n_parameters : int
    Number of trainable parameters in the model.

  Methods
  -------
  forward(x)
    Forward pass of the RNN classifier.
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
    print("Number of trainable parameters - RNN Classifier: ",
          self.n_parameters)

  def forward(self, x):
    """
    Perform a forward pass through the model.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape [batch_size, n_seq, input_size].

    Returns
    -------
    out : torch.Tensor
      Output tensor of shape [batch_size, n_classes] after passing through
      the RNN classifier.
    """

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


class BahdanauAttention(nn.Module):
  """
  Bahdanau Attention mechanism.

  Parameters
  ----------
  hidden_size : int
    The size of the hidden state in the attention mechanism.

  Methods
  -------
  forward(query, keys)
    Computes the context vector and attention weights.
  """

  def __init__(self, hidden_size):
    super(BahdanauAttention, self).__init__()
    self.wa = nn.Linear(hidden_size, hidden_size)
    self.ua = nn.Linear(hidden_size, hidden_size)
    self.va = nn.Linear(hidden_size, 1)

  def forward(self, query, keys):
    """
    Compute the context vector and attention weights.

    Parameters
    ----------
    query : torch.Tensor
      The query tensor of shape (batch_size, query_dim).
    keys : torch.Tensor
      The keys tensor of shape (batch_size, seq_len, key_dim).

    Returns
    -------
    context : torch.Tensor
      The context vector of shape (batch_size, 1, key_dim).
    weights : torch.Tensor
      The attention weights of shape (batch_size, 1, seq_len).
    """
    scores = self.va(torch.tanh(self.wa(query) + self.ua(keys)))
    scores = scores.squeeze(2).unsqueeze(1)

    weights = torch.nn.functional.softmax(scores, dim=-1)
    context = torch.bmm(weights, keys)

    return context, weights


class AttentionRNNClassifer(nn.Module):
  """
  AttentionRNNClassifer is a neural network model that combines an RNN with an
  attention mechanism for sequence classification tasks.

  Parameters
  ----------
  rnn_type : str
    Type of RNN to use (e.g., 'lstm', 'gru' or 'rnn').
  emb_size : int
    Size of the input embeddings.
  hidden_size : int
    Size of the hidden state in the RNN.
  n_classes : int
    Number of output classes.
  num_layers : int, optional
    Number of RNN layers (default is 1).
  batch_first : bool, optional
    If True, the input and output tensors are provided as (batch, seq, feature)
    (default is True).
  bidirectional : bool, optional
    If True, use bidirectional RNNs (default is False).
  linear_in_activation : nn.Module, optional
    Activation function to apply after the linear input layer
    (default is nn.Identity()).
  n_last_elements : int or None, optional
    Number of last elements to consider for mean pooling (default is 1).
  last_activation : str, optional
    Activation function to apply to the output ('sigmoid' or '')
    (default is 'sigmoid').

  Attributes
  ----------
  linear_in : nn.Linear
    Linear layer to transform input embeddings to hidden size.
  linear_in_activation : nn.Module
    Activation function applied after the linear input layer.
  rnn : nn.Module
    RNN layer.
  attention : BahdanauAttention
    Attention layer to compute context.
  linear_out : nn.Linear
    Linear layer to transform the concatenated context and RNN output to the
    number of classes.
  n_last_elements : int or None
    Number of last elements from the sequence to consider for mean pooling.
  last_activation : nn.Module
    Activation function applied to the output.
  n_parameters : int
    Number of trainable parameters in the model.

  Methods
  -------
  forward(x, hidden=None)
    Perform a forward pass through the model.
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
    print("Number of trainable parameters - RNN Classifier: ",
          self.n_parameters)

  def forward(self, x, hidden=None):
    """
    Perform a forward pass through the model.

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape [batch_size, n_seq, input_size].

    Returns
    -------
    out : torch.Tensor
      Output tensor of shape [batch_size, n_classes] after passing through
      the RNN classifier.
    """

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

  def __init__(self, model, classification_head, heads=None, student_projector=None, final_projection=None,
               label_embs_concat=None,
               num_classes=None, conf=None) -> None:
    super(ModelWrapper, self).__init__()

    self.model = model
    self.classification_head = classification_head

    self.center_C = None  # For MATPAC-style classification


    # For MATPAC-style classification
    self.student_projector = student_projector

    # For classic classification
    self.heads = heads

    self.conf = conf

    # For dynamic classification
    self.final_projection = final_projection
    self.label_embs_concat = label_embs_concat
    self.num_classes = num_classes

    if self.label_embs_concat is not None:
      nn.init.uniform_(self.label_embs_concat)

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
    if self.conf and self.conf.get("only_teacher_head", False):
      with torch.no_grad():
        emb = self.model(x)
      cla = None
      return cla, emb
    
    else:
      emb = self.model(x)
      cla = self.classification_head(emb)

      return cla, emb


if __name__ == "__main__":

  linear_att_head = LinearClassifer(emb_size=960,
                                    n_classes=527,
                                    pooling="attention",
                                    activation_att="sigmoid")

  linear_mean_head = LinearClassifer(emb_size=960,
                                     n_classes=527,
                                     pooling="mean")

  mlp_att_head = MLPClassifer(emb_size=960,
                              n_classes=527,
                              hidden_features_size=1280,
                              dropout=0.2,
                              pooling="mean")

  lstm_head = RNNClassifer(rnn_type="lstm",
                           emb_size=960,
                           hidden_size=256,
                           n_classes=527,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False,
                           last_activation="",)

  example = torch.rand((32, 10, 960))

  res_att = linear_att_head(example)
  print(res_att.shape)

  res_mean = linear_mean_head(example)
  print(res_mean.shape)

  res_mlp = mlp_att_head(example)
  print(res_mlp.shape)

  res_lstm = lstm_head(example)
  print(res_lstm.shape)
