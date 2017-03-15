# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 22:21:13 2017

@author: OKToffa
"""
import pickle as pkl
import lasagne
import theano
import numpy as np
#import matplotlib.pyplot as plt
from config import TRAINED_MODEL_FILE
import PIL.Image as Image

def predict(test_set_x, test_set_y, test_size):
    print('loading the trained model')
    #with open(TRAINED_MODEL_FILE, 'rb') as f:
        #u = pkl._Unpickler(f)
        #u.encoding = 'latin1'
        #classifier = u.load()    
    #classifier = pkl.load(open(TRAINED_MODEL_FILE, 'rb'), encoding='latin1')
    classifier = pkl.load(open(TRAINED_MODEL_FILE))
    print('compiling the predictor function')
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=lasagne.layers.get_output(classifier.network, deterministic=True))
    #test_set_x = test_set_x.get_value()
    predicted_values = predict_model(test_set_x[:test_size])
    
    print('printing the test values')
    #fig = plt.figure()
    for i in range(test_size):
        img_input = 255*test_set_y[i]
        img_input = img_input.astype('uint8')
        #plt.subplot(2, test_size, i+1)
        #plt.imshow(img_input)        
        img_output_temp = 255*predicted_values[i].reshape(3,64,64)
        img_output_temp = np.transpose(img_output_temp, (1, 2, 0))      
        img_output_temp = img_output_temp.astype('uint8')
        img_output = np.copy(img_input)
        for u in range(32):
            for v in range(32):
                img_output[u+16,v+16,:] = img_output_temp[u+16,v+16,:]         
        #plt.subplot(2, test_size, test_size+i+1)
        #plt.imshow(img_output)
        Image.fromarray(img_input).save('input%d.png'%i)
        Image.fromarray(img_output).save('output%d.png'%i)
    #plt.show()
    #plt.savefig("result.png", dpi=fig.dpi)
