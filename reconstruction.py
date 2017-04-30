# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 22:21:13 2017

@author: OKToffa
 Reconstruction inspired from:
[1]https://arxiv.org/pdf/1607.07539v2.pdf Semantic Image Inpainting with Perceptual and Contextual Losses
"""
import pickle as pkl
import lasagne
import theano
import theano.tensor as T
import numpy as np
from config import TRAINED_MODEL_FILE, OUTPUT_FOLDER
import os
from coco_loader import coco_loader
import lsgan
import PIL.Image as Image

def get_corruption_mask(batch_size):
    """
    returns the corruption mask
    """
    corupt_mask = np.ones(shape=(batch_size, 3, 64, 64)).astype('float32')
    center = (32,32)
    corupt_mask[:,:, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = 0
    return corupt_mask

def reconstruct(test_set_x, test_set_y, test_size, lbd = 0.0001, learning_rate=0.001, max_iteration=1000):
    print('loading the trained model')
    model = pkl.load(open(TRAINED_MODEL_FILE))
    disc = model[0]
    gen = model[1]
    
    # define input variables type
    original_image = test_set_x.get_value(borrow=False)
    corruption_mask = T.tensor4('corrution_mask')
    noise_shared = theano.shared(np.random.uniform(-1., 1., size=(test_size, 100)).astype(theano.config.floatX));
    print('compiling the generation function')
    reconstructed_image = lasagne.layers.get_output(gen, inputs=noise_shared, deterministic=True)

    print('compiling the contextual loss function')
    contextual_loss = lasagne.objectives.squared_error(reconstructed_image.flatten()* corruption_mask.flatten(), original_image.flatten()* corruption_mask.flatten()).mean()

    print('compiling the perceptual loss function')
    p_reconstructed = lasagne.layers.get_output(disc, inputs=reconstructed_image, deterministic=True) 
    perceptual_loss = lasagne.objectives.squared_error(p_reconstructed, T.ones(p_reconstructed.shape)).mean()
    
    print('compiling total loss and update function')
    reconstruction_loss = contextual_loss + lbd * perceptual_loss
    updates = lasagne.updates.adam(reconstruction_loss, [noise_shared], learning_rate = learning_rate) 
    reconstruction_fn = theano.function([corruption_mask], [reconstructed_image, reconstruction_loss],
        updates=updates)   
    
    print('printing target images')
    target_values = test_set_y.get_value(borrow=True)
    for i in range(test_size):
        img_target = 255*target_values[i]
        img_target = img_target.astype('uint8')       
        Image.fromarray(img_target).save('target%d.png'%i)   
    
    print('running the reconstruction batch')
    n_iter = 0
    print_freq=10
    corruption_mask = get_corruption_mask(test_size)
    while (n_iter < max_iteration):
        n_iter = n_iter + 1
        reconstruction_values, loss = reconstruction_fn(corruption_mask)
        if n_iter%print_freq == 0 or n_iter == 1:
            print("reconstruction loss %f"%loss)
            for i in range(test_size):
                img_output_temp = 255*reconstruction_values[i].reshape(3,64,64)
                img_output_temp = np.transpose(img_output_temp, (1, 2, 0))
                img_output_temp = img_output_temp.astype('uint8')
                Image.fromarray(img_output_temp).save('generated_img%d_iter%d.png'%(i,n_iter))
    
    
    for i in range(test_size):
        img_target = 255*target_values[i]
        img_target = img_target.astype('uint8')       
        img_output_temp = 255*reconstruction_values[i].reshape(3,64,64)
        img_output_temp = np.transpose(img_output_temp, (1, 2, 0))      
        img_output_temp = img_output_temp.astype('uint8')
        img_output = np.copy(img_target)
        for u in range(32):
            for v in range(32):
                img_output[u+16,v+16,:] = img_output_temp[u+16,v+16,:]         
        Image.fromarray(img_output_temp).save('generated_final%d.png'%i)
        Image.fromarray(img_output).save('output%d.png'%i)

if __name__ == '__main__':
    valset = coco_loader(split="val2014")
    #train_coco(dataset, valset, network_type=1)   
    batch_size = 5
    input_shared, target_shared, caption = valset.load_items(0, batch_size, transpose_y=False)
    os.chdir(OUTPUT_FOLDER)
    print("Predicting using the trained model ...")
    reconstruct(input_shared, target_shared, batch_size)
    os.chdir('../')
