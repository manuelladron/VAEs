import torch.nn as nn

activations = {
    'relu': nn.ReLU,
    'identity': nn.Identity,
    'elu': nn.ELU,
}
