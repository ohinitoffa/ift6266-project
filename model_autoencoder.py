# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 22:58:41 2017

@author: OKToffa
"""
import lasagne
from functools import reduce

class autoencoder(object):
    """Auto-Encoder class inspired from http://deeplearning.net/tutorial/dA.html"""

    def __init__(
        self,
        input=None,
        target=None,
        input_shape=(None, 3,64,64), 
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
    """Convolution Auto-Encoder classl"""

    def __init__(
        self,
        input=None,
        target=None,
        input_shape=(None,3,64,64), 
        n_hidden=3*32*32,
        n_convlayer=2,
        n_filter = 32,
        filter_sizex = 5,
        pool=2,
        nonlinearity = lasagne.nonlinearities.rectify
    ):
         # Parameters:
         #    input: Theano symbolic variable
         #    target: Theano symbolic variable
         #    input_shape : tuple of int or None elementsshape: tuple of int 
         #    n_convlayer : number of convolution and pooling layer.         
         #    n_hidden : int The number of units of the hidden layer
         #    nonlinearity : callable or None. applied to the activation layer.
  
        print("Init autoencoder network ...")
        self.input = input
        self.target = target
        
        #creating the encoder
        encoderLayer = lasagne.layers.InputLayer(shape=input_shape, input_var=input)
        # put some convolution layers with maxpooling
        image_size = input_shape[2]
        for i in range(n_convlayer):
            encoderLayer = lasagne.layers.Conv2DLayer(incoming = encoderLayer, num_filters=n_filter, filter_size=(filter_sizex, filter_sizex), nonlinearity=nonlinearity)
            encoderLayer = lasagne.layers.MaxPool2DLayer(incoming = encoderLayer, pool_size=(pool, pool))
            image_size = (image_size - filter_sizex + 1)/pool
        
        image_size = int(image_size)
        # put a full connected layer
        encoder_input_shape = (-1, n_filter, image_size, image_size) #lasagne.layers.get_output_shape(convlayer)     
        n_decoder_units = n_filter*image_size*image_size
        encoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_hidden, nonlinearity=nonlinearity)
        
        #creating the decoder
        decoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_decoder_units, W = encoderLayer.W.T, nonlinearity=nonlinearity)       
        
        #since the result of the decoderlayer is flatten, we have to unflatten it
        decoderLayer = lasagne.layers.ReshapeLayer(incoming = decoderLayer, shape = encoder_input_shape)
        # do the inverse operation of the convolution layers with maxunpooling
        for j in range(n_convlayer):
            decoderLayer = lasagne.layers.Upscale2DLayer(incoming = decoderLayer, scale_factor=pool)        
            decoderLayer = lasagne.layers.Deconv2DLayer(incoming = decoderLayer, num_filters=3, filter_size=(filter_sizex, filter_sizex), nonlinearity=nonlinearity)     
        self.network = decoderLayer
        
        
    def get_cost_updates(self):
        """ This function computes the cost and the updates for one training step """
        network_output = lasagne.layers.get_output(self.network)
        cost = lasagne.objectives.squared_error(network_output, self.target).mean()
        all_params = lasagne.layers.get_all_params(self.network,trainable=True)
        updates = lasagne.updates.adam(cost, all_params) #lasagne.updates.adadelta(cost, all_params)
        return (cost, updates)
    
def build_convautoencoder(
        input=None,
        input_shape=(None,3,64,64), 
        n_hidden=3*32*32,
        n_convlayer=2,
        n_filter = 32,
        filter_sizex = 5,
        pool=2,
        nonlinearity = lasagne.nonlinearities.rectify) :
    
        print("Init autoencoder network ...")
      
        #creating the encoder
        encoderLayer = lasagne.layers.InputLayer(shape=input_shape, input_var=input)
        # put some convolution layers with maxpooling
        image_size = input_shape[2]
        for i in range(n_convlayer):
            encoderLayer = lasagne.layers.Conv2DLayer(incoming = encoderLayer, num_filters=n_filter, filter_size=(filter_sizex, filter_sizex), nonlinearity=nonlinearity)
            encoderLayer = lasagne.layers.MaxPool2DLayer(incoming = encoderLayer, pool_size=(pool, pool))
            image_size = (image_size - filter_sizex + 1)/pool
        
        image_size = int(image_size)
        # put a full connected layer
        encoder_input_shape = (-1, n_filter, image_size, image_size) #lasagne.layers.get_output_shape(convlayer)     
        n_decoder_units = n_filter*image_size*image_size
        encoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_hidden, nonlinearity=nonlinearity)
        
        #creating the decoder
        decoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_decoder_units, nonlinearity=nonlinearity)       
        
        #since the result of the decoderlayer is flatten, we have to unflatten it
        decoderLayer = lasagne.layers.ReshapeLayer(incoming = decoderLayer, shape = encoder_input_shape)
        # do the inverse operation of the convolution layers with maxunpooling
        for j in range(n_convlayer):
            decoderLayer = lasagne.layers.Upscale2DLayer(incoming = decoderLayer, scale_factor=pool)        
            decoderLayer = lasagne.layers.Deconv2DLayer(incoming = decoderLayer, num_filters=3, filter_size=(filter_sizex, filter_sizex), nonlinearity=nonlinearity)     
        return decoderLayer    
    