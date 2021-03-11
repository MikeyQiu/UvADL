# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cpu'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        # Embedding shape: one hot embedding : vocabulary_size*vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, vocabulary_size, _weight=torch.eye(vocabulary_size))
        self.embedding.weight.requires_grad = False  # Don't learn embedding
        # Initialize network
        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers).to(device)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size).to(device)

    def forward(self, x, h, c):
        # Implementation here...
        x_embedding = self.embedding(x)
        # return result and intermediate state
        output, (hn, cn) = self.lstm(x_embedding, (h, c))
        output = self.linear(output)
        return output, hn, cn

    def complete_sentence(self, model, dataset, seq_length, tao=0, given_sentence=''):
        '''
        Compelete unfinshed sentence using generated model
        :param model: final model after training
        :param dataset: selected dataset
        :param seq_length: length of input sequence, e.g. T=30
        :param tao: temperature for random selection t={0.5,1,2}
        :param given_sentence: a string of unfinished sentence
        :return: a string generated sentence in the given length by model
        '''
        # convert char to ix
        ix_list = dataset.convert_to_index(given_sentence)
        char_list = [ix for ix in ix_list]

        h_0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_num_hidden).to(self.device)
        c_0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_num_hidden).to(self.device)
        current_char = torch.full((1, 1), ix_list[0], dtype=torch.long)
        for i in range(seq_length):
            output, h_n, c_n = model.forward(current_char, h_0, c_0)
            if tao == 0:  # deterministic
                next_char = output.argmax()
            else: # multinominal distribution
                output = (output * tao).squeeze(0)
                prob = torch.softmax(output, dim=1) # modify with tao
                next_char = torch.multinomial(prob, 1)
            if i < len(ix_list): # before given sentence being tranversed, select next character in the given sentence
                current_char = torch.full((1, 1), ix_list[i], dtype=torch.long)
            else: # update according to the newly generated character
                char_list.append(int(next_char))
                current_char = next_char.view(1, 1)
            # update the hidden state and cell
            h_0, c_0 = h_n, c_n
        strings = dataset.convert_to_string(char_list)
        return strings

    def generate_sequence(self, model, dataset, seq_length, tao=0):
        # Random first letter
        current_char = torch.randint(0, self.vocabulary_size, (1, 1))
        char_ix = int(current_char)
        char_list = []
        char_list.append(char_ix)
        h_0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_num_hidden).to(self.device)
        c_0 = torch.zeros(self.lstm_num_layers, 1, self.lstm_num_hidden).to(self.device)
        for i in range(seq_length - 1):
            output, h_n, c_n = model.forward(current_char, h_0, c_0)
            if tao == 0:  # deterministic
                next_char = output.argmax()
            else:
                output = (output * tao).squeeze(0)
                prob = torch.softmax(output, dim=1)
                next_char = torch.multinomial(prob, 1)
            char_list.append(int(next_char))
            # update the current_char
            current_char = next_char.view(1, 1)
            h_0, c_0 = h_n, c_n
        strings = dataset.convert_to_string(char_list)
        return strings
