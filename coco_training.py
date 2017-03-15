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
from config import TRAINING_EPOCHS, BATCH_SIZE, OUTPUT_FOLDER, NUM_HIDDEN, TRAINED_MODEL_FILE, INPUT_SHAPE
from model_autoencoder import autoencoder, convautoencoder
def train_coco(trainset, valset, network_type=0, training_epochs= TRAINING_EPOCHS, batch_size=BATCH_SIZE, 
          output_folder=OUTPUT_FOLDER):
    n_train_batches = len(trainset.imgs) // batch_size
    n_valid_batches = len(valset.imgs) // batch_size
    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995 
    validation_frequency = min(n_train_batches, patience // 2) 
    best_validation_cost = numpy.inf 
    #create output folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)    

    #variables type                    
    inputvar = T.tensor4('inputvar')
    targetvar = T.tensor4('targetvar')    
    
    if(network_type == 0):
        model = autoencoder(
            input=inputvar,
            target=targetvar,
            input_shape=INPUT_SHAPE, 
            n_hidden=NUM_HIDDEN
        ) 
    elif(network_type == 1):
        model = convautoencoder(
            input=inputvar,
            target=targetvar,
            input_shape=INPUT_SHAPE, 
            n_hidden=NUM_HIDDEN
        )         

    print("Building the model ...")
    cost, updates = model.get_cost_updates()
    train_model = theano.function(
        [inputvar,targetvar],
        cost,
        updates=updates,
       # givens={ x: inputvar, y : targetvar},        
        on_unused_input='ignore',
        mode='FAST_RUN'
    )
    
    validate_model = theano.function(
        inputs=[model.input],
        outputs=lasagne.layers.get_output(model.network, deterministic=True), 
        on_unused_input='ignore',
        mode='FAST_RUN'
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
            print("batch_index %d"%batch_index)
            input_shared, target_shared, caption = trainset.load_items(batch_index, batch_size)
            input, target = input_shared.get_value(borrow=True), target_shared.get_value(borrow=True)
            # changed shape from (batch_size,64,64,3)  to (batch_size,3,64,64)
            train_cost.append(train_model(input, target))
            print("trained %d"%batch_index)
            iter = (epoch - 1) * n_train_batches + batch_index
            if (iter + 1) % validation_frequency == 0:
                # compute loss on validation set
                validation_cost = []
                for valbatch_index in range(n_valid_batches):
                    valinput_shared, valtarget_shared, valcaption = valset.load_items(valbatch_index, batch_size)
                    valinput, valtarget = valinput_shared.get_value(borrow=True), valtarget_shared.get_value(borrow=True)
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
    
   