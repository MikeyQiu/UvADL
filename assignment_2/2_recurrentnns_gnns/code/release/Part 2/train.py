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
import os
import time
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
import random


# Inspired by tutorial:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
###############################################################################


def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    sentences = pd.DataFrame()
    loss_list = []
    accuracy_list = []
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)
    # vocabulary size
    data_size = dataset.vocab_size
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, data_size, config.lstm_num_hidden,
                                config.lstm_num_layers).to(device)
    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()
        #######################################################
        # Add more code here ...
        #######################################################
        batch_inputs = torch.stack(batch_inputs, dim=0)  # for input, by time step, vertical
        #initialize h_0 and c_0
        h_0 = torch.zeros(config.lstm_num_layers, batch_inputs.shape[1], config.lstm_num_hidden, device=config.device)
        c_0 = torch.zeros(config.lstm_num_layers, batch_inputs.shape[1], config.lstm_num_hidden, device=config.device)

        # forward propagation
        output, h_n, c_n = model(batch_inputs, h_0, c_0)
        batch_targets = torch.stack(batch_targets, dim=1).long()  # for validation, by sample, horizontal
        prediction = output.permute(1, 2, 0)  # transpose
        loss = criterion(prediction, batch_targets)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        prediction_value = torch.argmax(prediction, dim=1)  # one hot to value
        correct = (prediction_value == batch_targets).sum().item()
        accuracy = correct / (batch_targets.size(0) * batch_targets.size(1))# correct / all table's components

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        if (step + 1) % config.print_every == 0:
            print("[{}] Train Step {}/{}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

        if (step + 1) % config.sample_every == 0:
            generate_sentence = model.generate_sequence(model, dataset, 60, tao=2)
            pass

        if (step + 1) % config.save_every == 0:
            torch.save(model.state_dict(), model.__class__.__name__ + "_{}".format(step + 1) + ".pt")
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
            
    #2.1c Complete sentence
    for i in range(5):
        generate_sentence_00_60 = model.complete_sentence(model, dataset, 100, tao=0, given_sentence="Secretary of "
                                                                                                     "States")
        generate_sentence_05_60 = model.complete_sentence(model, dataset, 100, tao=0.5, given_sentence="Secretary of "
                                                                                                       "States")
        generate_sentence_10_60 = model.complete_sentence(model, dataset, 100, tao=1.0, given_sentence="Secretary of "
                                                                                                       "States")
        generate_sentence_20_60 = model.complete_sentence(model, dataset, 100, tao=2.0, given_sentence="Secretary of "
                                                                                                       "States")
        df = pd.DataFrame({'generate00': [generate_sentence_00_60],
                           'generate05': [generate_sentence_05_60],
                           'generate10': [generate_sentence_10_60],
                           'generate20': [generate_sentence_20_60],
                           })
        sentences = sentences.append(df)

    sentences.to_csv("sentences_complete.csv", index=False)
    # np.save("accuracy.npy", accuracy_list)
    # np.save("loss.npy", loss_list)
    torch.save(model.state_dict(), "model/" + model.__class__.__name__ + "_{}".format("final") + ".pt")
    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()
    # required = True,
    # Model params
    parser.add_argument('--txt_file', type=str, default="assets/book_EN_democracy_in_the_US.txt",
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')
    parser.add_argument('--device', type=str, default="cpu",
                        help="Training device 'cpu' or 'cuda:0'")
    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='DropoutT keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--save_every', type=int, default=2500,
                        help='How often to save hidden states from the model')
    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)
