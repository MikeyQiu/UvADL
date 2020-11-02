"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs=n_inputs
        self.n_hidden=n_hidden
        self.n_classes=n_classes
        self.depth=len(n_inputs)
        self.softmax_module=SoftMaxModule
        self.elu_module=ELUModule

        #raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        input=x
        for i in range(self.depth):
            self.linear_module = LinearModule(input,self.n_hidden[i])
            l_temp = self.linear_module.forward(input)
            h_temp = self.elu_module.forward(l_temp)
            input=h_temp

        out=self.softmax_module.forward(input)

        #raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        #raise NotImplementedError
        dx = self.softmax_module.backward(dout)
        for i in range(self.depth, 0, -1):
            self.linear_module = LinearModule(dx, self.n_hidden[i])
            h_temp = self.elu_module.backward(dx)
            l_temp = self.linear_module.backward(h_temp)
            dx = l_temp
        ########################
        # END OF YOUR CODE    #
        #######################

        return
