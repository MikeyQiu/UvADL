"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):
        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        ##Initialize embedding
        self.embedding_dim = 64
        self.embedding = nn.Embedding(3, self.embedding_dim)  # possible input3= [0,1,2](padding)
        self.embedding.weight.requires_grad = False  # Do not update weight
        ##Forget f##
        self.W_fx = nn.Parameter(torch.Tensor(self.embedding_dim, hidden_dim,device=self.device))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim,device=self.device))
        self.W_fb = nn.Parameter(torch.Tensor(hidden_dim,device=self.device))
        ##Update i##
        self.W_ix = nn.Parameter(torch.Tensor(self.embedding_dim, hidden_dim,device=self.device))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim,device=self.device))
        self.W_ib = nn.Parameter(torch.Tensor(hidden_dim,device=self.device))
        ##Add g##
        self.W_gx = nn.Parameter(torch.Tensor(self.embedding_dim, hidden_dim,device=self.device))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim,device=self.device))
        self.W_gb = nn.Parameter(torch.Tensor(hidden_dim,device=self.device))
        ##Output o##
        self.W_ox = nn.Parameter(torch.Tensor(self.embedding_dim, hidden_dim,device=self.device))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim,device=self.device))
        self.W_ob = nn.Parameter(torch.Tensor(hidden_dim,device=self.device))
        ##Predict p##
        self.W_ph = nn.Parameter(torch.Tensor(hidden_dim, num_classes,device=self.device))
        self.W_pb = nn.Parameter(torch.Tensor(num_classes,device=self.device))
        for weight in self.parameters():
            try:
                # weight
                nn.init.kaiming_normal_(weight)
            except:
                # bias
                nn.init.zeros_(weight)
        self.h_t = torch.zeros(hidden_dim, batch_size,device=self.device)
        self.c_t = torch.zeros(hidden_dim, batch_size,device=self.device)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        assert (self.batch_size, self.seq_length, self.input_dim) == x.shape
        # initialize the hidden states and cell
        h_t = self.h_t
        c_t = self.c_t
        for t in range(self.seq_length):
            x_t = x[:, t, :].long()  # 1,batch,1
            embed_x_t = self.embedding(x_t).squeeze(1) # change original input into embedding
            f_t = torch.sigmoid(embed_x_t @ self.W_fx + h_t @ self.W_fh + self.W_fb)
            i_t = torch.sigmoid(embed_x_t @ self.W_ix + h_t @ self.W_ih + self.W_ib)
            o_t = torch.sigmoid(embed_x_t @ self.W_ox + h_t @ self.W_oh + self.W_ob)
            g_t = torch.tanh(embed_x_t @ self.W_gx + h_t @ self.W_gh + self.W_gb)
            c_t = f_t * c_t + i_t * g_t
            h_t = torch.tanh(c_t) * o_t
        # make prediction
        p_t = (h_t @ self.W_ph + self.W_pb).squeeze()
        y_predict = torch.log_softmax(p_t, dim=1)  # Log softmax+NLLLoss=CrossEntropy
        return y_predict
        ########################
        # END OF YOUR CODE    #
        #######################
