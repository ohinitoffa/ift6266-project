# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 13:50:55 2017

@author: OKToffa
"""
from __future__ import print_function

import os
import timeit
import numpy
import lasagne
import theano
import theano.tensor as T
import pickle as pkl
from config import TRAINING_EPOCHS, BATCH_SIZE, OUTPUT_FOLDER, NUM_HIDDEN, TRAINED_MODEL_FILE
from model_autoencoder import autoencoder
def train_coco(trainset, valset, training_epochs= TRAINING_EPOCHS, batch_size=BATCH_SIZE, 
          output_folder=OUTPUT_FOLDER):
    n_train_batches = len(trainset.imgs) // batch_size
    n_valid_batches = len(valset.imgs) // batch_size
    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995 
    validation_frequency = min(n_train_batches, patience // 2) 
    best_validation_cost = numpy.inf 

    #variables type                    
    x = T.tensor4('x')
    y = T.tensor4('y')
    inputvar = T.tensor4('inputvar')
    targetvar = T.tensor4('targetvar')    
    
    model = autoencoder(
        input=x,
        target=y,
        input_shape=(None,3,64,64), 
        n_hidden=NUM_HIDDEN
    )    

    print("Building the model ...")
    cost, updates = model.get_cost_updates()
    train_model = theano.function(
        [inputvar,targetvar],
        cost,
        updates=updates,
        givens={ x: inputvar, y : targetvar},        
        on_unused_input='ignore'
    )
    
    validate_model = theano.function(
        inputs=[model.input],
        outputs=lasagne.layers.get_output(model.network), 
        on_unused_input='ignore'
    )     

    start_time = timeit.default_timer()

    print("Training the model ...")
    print('Epochs ', training_epochs)
    print('Batches number ', n_train_batches)
    print('Batches size ', batch_size)
    
    end_loop = False
    epoch = 0
    while (epoch < training_epochs) and (not end_loop):
        epoch = epoch + 1    
        # go through training set
        train_cost = []
        for batch_index in range(n_train_batches):
            input, target, caption = trainset.load_items(batch_index, batch_size)
            # changed shape from (batch_size,64,64,3)  to (batch_size,3,64,64)
            train_cost.append(train_model(input.transpose((0, 3, 1, 2)), target.transpose((0, 3, 1, 2))))
            iter = (epoch - 1) * n_train_batches + batch_index
            if (iter + 1) % validation_frequency == 0:
                # compute loss on validation set
                validation_cost = []
                for valbatch_index in range(n_valid_batches):
                    valinput, valtarget, valcaption = valset.load_items(valbatch_index, batch_size)
                    valinput = valinput.transpose((0, 3, 1, 2))
                    valtarget = valtarget.transpose((0, 3, 1, 2))
                    network_output = validate_model(valinput)
                    val_loss = ((network_output.flatten() - valtarget.flatten())**2).mean()
                    validation_cost.append(val_loss) 
                this_validation_cost = numpy.mean(validation_cost, dtype='float64')
                print(
                    'epoch %i, batch %i/%i, validation error %f' %
                    ( epoch, batch_index + 1, n_train_batches,this_validation_cost)
                )

                # if we got the best validation score until now
                if this_validation_cost < best_validation_cost:
                    #improve patience if loss improvement is good enough
                    if this_validation_cost < (best_validation_cost * improvement_threshold):
                        patience = max(patience, iter * patience_increase)
                    best_validation_cost = this_validation_cost
                    os.chdir(output_folder)    
                    with open(TRAINED_MODEL_FILE, 'wb') as f:
                        pkl.dump(model, f)
                    os.chdir('../')
            if patience <= iter:
                end_loop = True
                break
        print('Training epoch %d, cost ' % epoch, numpy.mean(train_cost, dtype='float64'))
    end_time = timeit.default_timer()
    training_time = (end_time - start_time) 

    print('The training ran for %.2fm' % ((training_time) / 60.))
    
   