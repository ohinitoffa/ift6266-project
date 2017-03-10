# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 22:58:41 2017

@author: OKToffa
"""
import lasagne

class autoencoder(object):
    """Auto-Encoder class inspired from http://deeplearning.net/tutorial/dA.html"""

    def __init__(
        self,
        input=None,
        target=None,
        input_shape=(None, 3*64*64), 
        n_hidden=3*32*32,
        nonlinearity = lasagne.nonlinearities.rectify
    ):
         # Parameters:
         #    input: Theano symbolic variable
         #    target: Theano symbolic variable
         #    input_shape : tuple of int or None elementsshape: tuple of int 
         #    n_hidden : int The number of units of the hidden layer
         #    nonlinearity : callable or None. applied to the activation layer.
  
        print("Init autoencoder network ...")
        self.input = input
        self.target = target
        n_input = reduce(lambda x, y: x*y, input_shape[1:])
        inputLayer = lasagne.layers.InputLayer(shape=input_shape, input_var=input)
        encoderLayer = lasagne.layers.DenseLayer(incoming = inputLayer, num_units = n_hidden,W = lasagne.init.Normal(), nonlinearity=nonlinearity)
        self.encoderLayer = encoderLayer
        decoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_input, W = encoderLayer.W.T, nonlinearity=nonlinearity)       
        self.network = decoderLayer
        
        
    def get_cost_updates(self):
        """ This function computes the cost and the updates for one training step """
        network_output = lasagne.layers.get_output(self.network)
        cost = lasagne.objectives.squared_error(network_output.flatten(), self.target.flatten()).mean()
        all_params = lasagne.layers.get_all_params(self.network,trainable=True)
        updates = lasagne.updates.adam(cost, all_params) #lasagne.updates.adadelta(cost, all_params)
        return (cost, updates)
    
class convautoencoder(object):
    """Convolution Auto-Encoder class"""

    def __init__(
        self,
        input=None,
        input_shape=(None, 3, 64, 64), 
        n_hidden=3*32*32,
        nonlinearity = lasagne.nonlinearities.rectify
    ):
         # Parameters:
         #    input: Theano symbolic variable
         #    input_shape : tuple of int or None elementsshape: tuple of int 
         #    n_hidden : int The number of units of the hidden layer
         #    nonlinearity : callable or None. applied to the activation layer. 
             
        print("Init autoencoder network ...")
        self.input = input
        n_input = reduce(lambda x, y: x*y, input_shape[1:])
        inputLayer = lasagne.layers.InputLayer(shape=input_shape, input_var=input)
        encoderLayer = lasagne.layers.DenseLayer(incoming = inputLayer, num_units = n_hidden, nonlinearity=nonlinearity)
        decoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_input, W=encoderLayer.W.T, nonlinearity=nonlinearity)
        self.network = decoderLayer
        
        
    def get_cost_updates(self):
        """ This function computes the cost and the updates for one training step """
        network_output = lasagne.layers.get_output(self.network)
        cost = lasagne.objectives.binary_crossentropy(network_output, input).mean()
        all_params = lasagne.layers.get_all_params(self.network,trainable=True)
        updates = lasagne.updates.adadelta(cost, all_params)
        return (cost, updates)