"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = None
        self.grads=None
        mean=0
        std=0.0001
        self.params['weight']=np.random.normal(loc=mean, scale=std,size=(out_features,in_features))
        self.params['bias']=np.zeros(out_features)
        self.params['gradient']=np.zeros(out_features)
        #raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out=np.dot(self.params["weights"],x)+self.params["bias"]
        self.params["linear_cache"] = (x, self.params["weights"], self.params["bias"])

        #raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        x_prev, W, b = self.params["linear_cache"]
        # print(np.shape(dZ))
        m = x_prev.shape[1]

        # YOUR CODE HERE
        dW = 1 / m * np.dot(dout, x_prev.T)
        db = 1 / m * np.sum(dout, axis=1, keepdims=True)
        # print(np.shape(db))
        dx = np.dot(W.T, dout)
        self.grads['weight']=dW
        self.grads['bias']=db
        # YOUR CODE ENDS HERE
        # YOUR CODE ENDS HERE
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        #raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        b = x.max()
        y = np.exp(x - b)
        self.out = y / y.sum()
        #raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
        return self.out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dbout = np.diag(self.out) - np.outer(self.out, self.out)
        dx= np.dot(dout, dbout)
        #raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        m = y.shape[1]
        out = -1 / m * np.sum(np.log(x) + (1 - y) * np.log(1 - x))
        #raise NotImplementedError

        ########################
        # END OF YOUR CODE    #
        #######################

        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = - (np.divide(y, x) - np.divide(1 - y, 1 - x))
        #raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = x
        if x>0:
            self.out=x
        else:
            self.out=np.exp(x)-1

        #raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return self.out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if self.input>0:
            delu=1
        else:
            delu=np.exp(self.input)
        dx = np.dot(delu, dout)
        #raise NotImplementedError

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
