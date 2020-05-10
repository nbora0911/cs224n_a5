#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch

class CNN(torch.nn.Module):
    """
    what does this module do?
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        """
        @param in_channels (int): dimension of character embeddings
        @param out_channels (int): number of output features = the size of the window used to compute features
        @param kernel_size (int): kernel size

        """
        
        super(CNN, self).__init__()
        self.covn1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        """
        @param x (Tensor): reshaped input  tensor (batch_size, char_embed_dim, max_word_length)
        @oaram x_conv_out (Tensor): Tensor of shape (batch_size, f)
        """

        x_conv = self.covn1d(x)
        # x_conv is of size (f, max_word_length -k +1)
        x_conv_out = torch.max(torch.nn.functional.relu(x_conv), dim=2)[0]

        return x_conv_out

### END YOUR CODE
