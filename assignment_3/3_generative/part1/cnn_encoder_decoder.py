################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network

        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            latent_dim - Dimensionality of latent representation z
        """
        super().__init__()

        c_hid = num_filters
        act_fn = nn.GELU
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 28*28 => 14*14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 14x14 => 7x7
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 7x7 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
        )
        self.mean = nn.Linear(2 * 16 * c_hid, z_dim)
        self.log_std = nn.Linear(2 * 16 * c_hid, z_dim)
        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.
        # raise NotImplementedError

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        # convert input to hidden dimension
        h = self.encoder(x)
        # generate mean and std as latent variable z separately
        mean = self.mean(h)
        log_std = self.log_std(h)
        return mean, log_std


class CNNDecoder(nn.Module):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.
        Inputs:
            - num_input_channels: Number of channels of the image to
                                  reconstruct. For MNIST, this parameter is 3
            - num_filters: Number of filters we use in the last convolutional
                           layers. Early layers might use a duplicate of it.
            - latent_dim: Dimensionality of latent representation z
        """
        super().__init__()
        c_hid = num_filters
        act_fn = nn.GELU
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            # output = (input - 1)*stride+output_padding - 2*padding+kernel
            # output=(3*2)+out_padding-2+3=7
            # =>out_padding=0
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=0, padding=1, stride=2),
            # out_padding
            # 4x4 => 7*7 # convert to 7*7, output_padding<stride
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 7*7 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 14x14 => 28*28
            # nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is 
        # sufficient for the assignment.
        # raise NotImplementedError

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
