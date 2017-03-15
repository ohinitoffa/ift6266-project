# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 21:48:25 2017

@author: OKToffa
"""
from __future__ import print_function
import os
from config import OUTPUT_FOLDER
from coco_loader import coco_loader
from coco_training import train_coco
from predict import predict

if __name__ == '__main__':
    #dataset = coco_loader(split="train2014")
    valset = coco_loader(split="val2014")
    #train_coco(dataset, valset, network_type=1)   
    input, target, caption = valset.load_items(0, 10, transpose_y=False)
    test_set_x, test_set_y = input.get_value(borrow=True), target.get_value(borrow=True)
    os.chdir(OUTPUT_FOLDER)
    print("Predicting using the trained model ...")
    predict(test_set_x, test_set_y, 10)
    os.chdir('../')