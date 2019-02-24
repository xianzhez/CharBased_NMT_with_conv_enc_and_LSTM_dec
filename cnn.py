#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, m_word_size, e_char_size, f_filters):
        """
        @param m_word_size: hyper parameter
        @param e_char_size: hyper parameter
        @param f_filters: hyper parameter, output size, equals to e_word_size
        """
        super(CNN, self).__init__()
        k_kernel_size = 5
        # self.conv1d = nn.Conv1d((e_char_size, k_kernel_size), (f_filters, m_word_size - k_kernel_size +1), k_kernel_size)
        self.conv1d = nn.Conv1d(e_char_size, f_filters, k_kernel_size)
        self.maxpool = nn.MaxPool1d(m_word_size - k_kernel_size +1)
    
    def forward(self, x_reshaped):
        """
        @param x_reshaped: shape(batch_size, e_char_size, m_word_size)
        @return x_conv_out: shape(batch_size, f_filters)
        """
        # shape(batch_size, e_char_size, m_word_size - k_kernel_size +1)
        # print("x_reshaped = ", x_reshaped.size())
        x_conv = self.conv1d(x_reshaped)
        # print("x_conv = ", x_conv.size())
        # x_conv_out = torch.max(F.relu(x_conv), dim=2)
        x_conv_relu = F.relu(x_conv)
        # pooling k = m_word_size - k_kernel_size +1 = 21 - 5 + 1
        x_conv_out = F.max_pool1d(x_conv_relu, 17)
        return x_conv_out

### END YOUR CODE

