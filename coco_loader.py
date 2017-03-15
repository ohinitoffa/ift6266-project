# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 00:45:16 2017

@author: OKToffa
"""
import os
import glob
import pickle as pkl
import numpy as np
import theano
import PIL.Image as Image
class coco_loader(object):

    def __init__(
        self,
        mscoco="inpainting/", 
        split="train2014", 
        caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"
    ):
         # Parameters:
         #    mscoco: string coco folder
         #    split : string training folder
         #    caption_path: string caption path
        
        print('Loading ' + split + ' data...')
        self.mscoco = mscoco
        self.data_path = os.path.join(mscoco, split) 
        self.imgs = glob.glob(self.data_path + "/*.jpg")
        caption_path = os.path.join(mscoco, caption_path)
        with open(caption_path, 'rb') as fd:
            caption_dict = pkl.load(fd)
        self.caption_dict = caption_dict 
        self.x = np.array(0)
        self.y = np.array(0)
        
    def load_items(self, batch_idx, batch_size, transpose_x=True, transpose_y=True):     
        batch_imgs = self.imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]         
        res = [self.load_item(i, img_path) for i, img_path in enumerate(batch_imgs)]
        #remove None and unzip the list
        self.x, self.y, cap = zip(*[x for x in res if x is not None])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        if(transpose_x):
            self.x = self.x.transpose((0, 3, 1, 2))
        if(transpose_y):
            self.y = self.y.transpose((0, 3, 1, 2)) 
        return theano.shared(np.array(self.x), borrow = True), theano.shared(np.array(self.y), borrow = True), cap
        
    def load_item(self, index, img_path):     
        img = Image.open(img_path)
        img_array = np.array(img)
        target = img_array       
        cap_id = os.path.basename(img_path)[:-4]
        
        # create 32x32 black squre in the middle of the image
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
        else:
            # skip gray images
            return None
            #return the normalized values
        return input.astype('float32')/255., target.astype('float32')/255, self.caption_dict[cap_id]
                