# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:11:03 2017

@author: OKToffa
    Deep Convolution Generative Adversarial Network inspired from:
    [1]https://arxiv.org/pdf/1511.06434.pdf UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
    [2]https://arxiv.org/pdf/1406.2661.pdf Generative Adversarial Nets
    [3]https://arxiv.org/pdf/1701.00160.pdf NIPS 2016 Tutorial: Generative Adversarial Networks
    [4]https://github.com/soumith/ganhacks
    [5]https://arxiv.org/pdf/1607.07539v2.pdf Semantic Image Inpainting with Perceptual and Contextual Losses
    [6]https://arxiv.org/pdf/1611.07004v1.pdf Image-to-Image Translation with Conditional Adversarial Networks
    [7]https://arxiv.org/pdf/1611.04076v2.pdf Least Squares Generative Adversarial Networks
"""
import timeit
import numpy as np
import lasagne
import theano
import theano.tensor as T
import pickle as pkl
import os
import PIL.Image as Image
from coco_loader import coco_loader
from config import TRAINING_EPOCHS, BATCH_SIZE, OUTPUT_FOLDER, TRAINED_MODEL_FILE
class lsgan(object):
        def __init__(
        self):
            print("Init lsgan network ...")
        
        def build_generator(
                self,
                input_var=None,
                input_shape=(None, 100,1,1), 
                n_units=1024):
        		  #size 100
                network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
        		  # size 4x4x1024
                network = lasagne.layers.BatchNormLayer(incoming=network)
                network = lasagne.layers.Deconv2DLayer(incoming=network, num_filters=n_units, filter_size=(4, 4), stride=(1, 1))
                network = lasagne.layers.BatchNormLayer(incoming=network)
        		  # size 8x8x512: give 9x9 instead
                n_units = n_units//2
                network = lasagne.layers.Deconv2DLayer(incoming=network, num_filters=n_units,filter_size=(5, 5), stride=(2, 2), crop=1) 
        		  # size 16x16x256: give 17x17
                n_units = n_units//2
                network = lasagne.layers.BatchNormLayer(incoming=network)
                network = lasagne.layers.Deconv2DLayer(incoming=network, num_filters=n_units,filter_size=(5, 5), stride=(2, 2), crop=2)  
        		  # size 32x32x128: give 32x32
                n_units = n_units//2
                network = lasagne.layers.BatchNormLayer(incoming=network)
                network = lasagne.layers.Deconv2DLayer(incoming=network, num_filters=n_units,filter_size=(5, 5), stride=(2, 2), crop=2)                  
                # size 64x64: no batch normalization for the output layer as required in [1]
                network = lasagne.layers.Deconv2DLayer(incoming=network, num_filters=3, filter_size=(4, 4), stride=(2, 2), crop=2, 
                                                       nonlinearity=lasagne.nonlinearities.tanh)        
                return network
            
        def build_discriminator(
                self,
                input_var=None,
                input_shape=(None, 3, 64, 64), 
                n_units=512):
                # In the LeakyReLU, the slope of the leak was set to 0.2
                nonlinearity = lasagne.nonlinearities.LeakyRectify(0.2) 
                network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
                #size 32x32x64 no batch nomralization for the input as required in [1]
                network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=n_units//8, filter_size=(5, 5), stride=2, pad=2, nonlinearity=nonlinearity)
                #size 16x16x128
                network = lasagne.layers.BatchNormLayer(incoming=network)
                network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=n_units//4, filter_size=(5, 5), stride=2, pad=2, nonlinearity=nonlinearity)
                #size 8x8x256
                network = lasagne.layers.BatchNormLayer(incoming=network)
                network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=n_units//2, filter_size=(5, 5), stride=2, pad=2, nonlinearity=nonlinearity)                
                #size 4x4x512
                network = lasagne.layers.BatchNormLayer(incoming=network)
                network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=n_units, filter_size=(5, 5), stride=2, pad=2, nonlinearity=nonlinearity)

                network = lasagne.layers.FlattenLayer(incoming=network)
                network = lasagne.layers.DenseLayer(incoming=network, num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
                return network   
            
def gen_theano_fn(lbd=0.0001, reconstruction_lbd = 0.0001, reconstruction_lr=0.001):
    """
    Build the networks and returns the train functions
    """

    # input variables type
    input_noise = T.tensor4('input_noise')
    input_image = T.tensor4('input_image')
    corruption_mask = T.tensor4('corruption_mask')
    
    # Shared variable for image reconstruction
    #reconstruction_noise_shared = theano.shared(np.random.normal(1., 1., size=(1, 100)).astype(theano.config.floatX))

    # Build generator and discriminator
    dc_gan = lsgan()
    generator = dc_gan.build_generator(input_var=None)
    discriminator = dc_gan.build_discriminator(input_var=None)

    # Get output images from generator. The deterministic one is used for prediction
    image_fake = lasagne.layers.get_output(generator, inputs=input_noise)
    image_fake_det = lasagne.layers.get_output(generator, inputs=input_noise, deterministic=True)
    #reconstructed_image = lasagne.layers.get_output(generator, inputs=reconstruction_noise_shared, deterministic=True)

    # Get output probabilities from discriminator. The deterministic one is used for prediction
    p_real = lasagne.layers.get_output(discriminator, inputs=input_image)
    p_fake = lasagne.layers.get_output(discriminator, inputs=image_fake)
    p_fake_det = lasagne.layers.get_output(discriminator, inputs=image_fake_det, deterministic=True)
    #p_reconstructed = lasagne.layers.get_output(discriminator, inputs=reconstructed_image, deterministic=True)    

    # Compute loss for discriminator using least square [7] by minimizing the error on true and fake images
    d_loss_real = lasagne.objectives.squared_error(p_real, T.ones(p_real.shape)).mean()
    d_loss_fake = lasagne.objectives.squared_error(p_fake, T.zeros(p_fake.shape)).mean()    
    loss_discr = d_loss_real + d_loss_fake

    # Compute loss for generator using least square [7] by minimizing the error on fake images
    loss_gener_fake = lasagne.objectives.squared_error(p_fake, T.ones(p_fake.shape)).mean()   
    
    # Add L1 loss as in [6] using the contour
    l1_contour_loss = lasagne.objectives.squared_error(input_image* corruption_mask, image_fake* corruption_mask).mean()
    loss_gener = loss_gener_fake + lbd*l1_contour_loss

    # Gets the params dict for discriminator and generator
    params_discr = lasagne.layers.get_all_params(discriminator, trainable=True)
    params_gener = lasagne.layers.get_all_params(generator, trainable=True)

    # Update rules
    updates_discr = lasagne.updates.adam(loss_discr, params_discr, learning_rate=0.0002, beta1=0.5)
    updates_gener = lasagne.updates.adam(loss_gener, params_gener, learning_rate=0.0002, beta1=0.5)
    
    # Compute contextual and perceptual loss as in [5]
    #contextual_loss = T.mean(T.abs_(reconstructed_image * corruption_mask - input_image * corruption_mask))
    #perceptual_loss = lasagne.objectives.squared_error(p_reconstructed, T.ones(p_reconstructed.shape)).mean()       
    #reconstruction_loss = contextual_loss + reconstruction_lbd * perceptual_loss 
    #reconstruction_grad = T.grad(reconstruction_loss, reconstruction_noise_shared) 
    #updates_reconstruction = reconstruction_noise_shared - reconstruction_lr * reconstruction_grad 

    # Compile Theano functions
    train_d = theano.function( [input_image, input_noise], loss_discr, updates=updates_discr)
    train_g = theano.function( [input_image, input_noise, corruption_mask], loss_gener, updates=updates_gener)
    predict = theano.function([input_noise], [image_fake_det, p_fake_det])
    #reconstruction_f = theano.function([input_image, corruption_mask], [reconstruction_noise_shared, reconstructed_image, reconstruction_loss, reconstruction_grad],
        #updates=[(reconstruction_noise_shared, updates_reconstruction)])    

    return train_d, train_g, predict, (discriminator, generator)

def save_array_sample(imgarray, img_size):
    print('saving samples')
    for i in range(img_size):
        sample = 255*imgarray[i]
        sample = sample.astype('uint8')      
        sample = np.transpose(sample, (1, 2, 0))      
        Image.fromarray(sample).save('sample%d.png'%i)            
        
def train_coco(trainset, training_epochs= TRAINING_EPOCHS, batch_size=BATCH_SIZE,
               noise_size = 100, lr = 0.0002, output_folder=OUTPUT_FOLDER):
    n_train_batches = len(trainset.imgs) // batch_size
    gen_freq = 10                 

    print("Building and Compiling the model ...")
    theano_fn = gen_theano_fn()
    disc_train_model, gen_train_model, predict, model = theano_fn    
    
    start_time = timeit.default_timer()

    print("Training the model ...")
    print('Epochs ', training_epochs)
    print('Batches number ', n_train_batches)
    print('Batches size ', batch_size)
    
    epoch = 0
    while (epoch < training_epochs):
        epoch = epoch + 1    
        # go through training set
        disc_cost = []
        gen_cost = []
        for batch_index in range(n_train_batches):
            mask_shared, target_shared, caption = trainset.load_items(batch_index, batch_size, use_input_mask=True)
            corr_mask = mask_shared.get_value(borrow=True)
            target = target_shared.get_value(borrow=True)
            noise =  np.random.normal(1.,1,size=(target.shape[0], noise_size, 1, 1)).astype(theano.config.floatX) 
            #input = np.random.normal(1., 1, size=(batch_size, 3, 32, 32)).astype('float32')
            print(noise.shape)
            print(corr_mask.shape)
            if batch_index % gen_freq == 0:
                gen_cost.append(gen_train_model(target, noise, corr_mask))
            else:
                disc_cost.append(disc_train_model(target, noise))
            print("trained %d"%batch_index)
        #create output folder and generate samples
        epoch_folder = "result_epoch_%d"%epoch
        if not os.path.isdir(epoch_folder):        
            os.makedirs(epoch_folder)         
        os.chdir(epoch_folder)
        sample_noise = np.random.uniform(-1., 1., size=(batch_size, noise_size, 1, 1)).astype(theano.config.floatX)
        imgs_noise, probs_noise = predict(sample_noise)   
        save_array_sample(imgs_noise, batch_size)
        
        #save the samples and save the model
        with open(TRAINED_MODEL_FILE, 'wb') as f:
            pkl.dump(model, f)        
        os.chdir('../')
        print('Training epoch %d, disc cost %.2f, gen cost %.2f  '%( epoch, np.mean(disc_cost, dtype='float64'), np.mean(gen_cost, dtype='float64')))
    end_time = timeit.default_timer()
    training_time = (end_time - start_time) 

    print('The training ran for %.2fm' % ((training_time) / 60.))    
    
if __name__ == '__main__':  
    dataset = coco_loader(split="train2014") 
    train_coco(dataset)   
    
    #x = T.ftensor4()
    #dc_gan = lsgan()
    #h  = dc_gan.build_generator(input_var = x, input_shape=(2, 100,1,1))   
    #f = theano.function([x], lasagne.layers.get_output(h))
    #noise = np.random.normal(size=(2,100,1,1)).astype('float32')
    #n = f(noise)
    #print(n.shape)   
    
       
