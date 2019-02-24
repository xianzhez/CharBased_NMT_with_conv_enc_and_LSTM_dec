#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, e_word_size):
        """
        @param e_word_size : size of output word
        
        """
        super(Highway, self).__init__()
        self.e_word_size = e_word_size
        self.x_projection = nn.Linear(e_word_size, e_word_size, bias=True)
        # relu
        self.x_gate = nn.Linear(e_word_size, e_word_size, bias=True)
        # sigmoid
    
    def forward(self, x_conv_outs):
        """
        # @param x_conv_outs (tensor): shape(batch_size, max_sentence_length, max_word_length, e_word_size)
        @param x_conv_outs (tensor): shape(words_batch_size, e_word_size)
        # @return x_highways: shape(batch_size, max_sentence_length, max_word_length, e_word_size)
        @return x_highways: shape(words_batch_size, e_word_size)
        """
        # print("Highway forward x_conv_outs = ", x_conv_outs.size())
        # print("Highway forward self.x_projection = ", self.x_projection.weight.size())
        x_conv_outs = x_conv_outs.permute(0,2,1)
        # x_conv_outs = x_conv_outs.permute(1,0)
        x_projs_tmp = self.x_projection(x_conv_outs)
        x_projs = F.relu(x_projs_tmp)
        x_gates_tmp = self.x_gate(x_conv_outs)
        x_gates = F.sigmoid(x_gates_tmp)

        x_highways = x_gates * x_projs + (1-x_gates) * x_conv_outs
        return x_highways

### END YOUR CODE 

