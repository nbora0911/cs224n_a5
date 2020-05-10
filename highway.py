#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
### YOUR CODE HERE for part 1d
class Highway(torch.nn.Module):
    """ Highway Netowrk: This feedforward neural network contains two linear layers
    - Projection Layer
    - Gate Layer 
    """
    def __init__(self, out_channels):
        """
        @param x_covn_out_dim (int): size of the output of convolutional layer
        """
        super(Highway, self).__init__()

        self.x_proj_layer = torch.nn.Linear(out_channels, out_channels, bias=True)
        self.x_gate_layer = torch.nn.Linear(out_channels, out_channels, bias=True)
    
    def forward(self, x):
        """
        @param x (Tensor): input tensor obtained from covnet's output (batch_size, out_channels)
        @return x_highway (Tensor): output tensor from highway network - same size as input (batch_size, out_channels)
        """
        x_proj = self.x_proj_layer(x)
        x_gate = self.x_gate_layer(x)
        x_proj_relu = torch.relu(x_proj)
        x_gate_sigmoid = torch.sigmoid(x_gate) 
        x_highway = x_proj_relu*x_gate_sigmoid + (1-x_gate_sigmoid)*x

        return x_highway

### END YOUR CODE 

