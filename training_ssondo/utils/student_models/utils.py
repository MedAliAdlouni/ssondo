"""Utility functions for machine listening tasks."""


def count_parameters(model):
  """Simple function to count the parameters of a torch model

  Parameters
  ----------
  model : nn.module
      The model for which we want to compute the number of parameters

  Returns
  -------
  int
      Number of trainable parameters
  """
  total_param = 0
  for name, param in model.named_parameters():
    if param.requires_grad:
      num_param = param.numel()
      total_param += num_param
  return total_param
