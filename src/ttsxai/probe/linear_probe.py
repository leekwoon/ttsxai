"""
    adapted from:
    * https://github.com/fdalvi/NeuroX/blob/master/neurox/interpretation/linear_probe.py
"""
import torch.nn as nn


class LinearProbe(nn.Module):
    """Torch model for linear probe"""

    def __init__(self, input_size, num_classes):
        """Initialize a linear model"""
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """Run a forward pass on the model"""
        out = self.linear(x)
        return out