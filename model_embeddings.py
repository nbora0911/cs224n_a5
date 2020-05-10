#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        self.char_embed_size = 50
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embedding_dim=self.char_embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(in_channels=self.embeddings.embedding_dim, out_channels=embed_size)
        self.highway = Highway(out_channels=self.cnn.covn1d.out_channels)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        sentence_length = input_tensor.shape[0]
        batch_size = input_tensor.shape[1]
        max_word_length = input_tensor.shape[2]
        char_embed_size = self.embeddings.embedding_dim
        embed_dim = self.cnn.covn1d.out_channels

        output_embed = self.embeddings(input_tensor, device=self.device) # shape: (sentence_length, batch_size, max_word_length, e_char)
        output_embed_permute = output_embed.permute(0,1,3,2) # shape: (sentence_length, batch_size, e_char, max_word_length)
        output_embed_reshape = output_embed_permute.view(-1, char_embed_size, max_word_length)
        
        output_conv_net = self.cnn.forward(output_embed_reshape)   #(batch_size, char_embed_dim, max_word_length)
        output_highway = self.highway.forward(output_conv_net)  #shape: (batch_size*sentence_length, e_word)
        output = self.dropout(output_highway.view(sentence_length, batch_size, embed_dim)) #shape:  (sentence_length, batch_size, e_word)
        return output