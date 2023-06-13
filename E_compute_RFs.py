#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:20:16 2023

@author: pokor076
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import json
model = tf.keras.applications.vgg16.VGG16()
rf_per_layer = []
for n in np.arange(1,20):
    layers = np.arange(1,n)
    # get strides for all layers
    strides = []
    for layer in layers:
        conv_layer = model.layers[layer]
        stride = conv_layer.strides
        strides.append(stride[0])

    rf = []
    for layer in layers:
        conv_layer = model.layers[layer]
        try:
            kernel_size = conv_layer.kernel_size
        except:
            kernel_size = conv_layer.pool_size
    #formula from Araujo
        rf.append((kernel_size[0] - 1) * np.prod(strides[0:(layer-1)]))
    rf_per_layer.append(sum(rf)+1)

#plot all
pyplot.plot(np.arange(1,20),rf_per_layer,
            linestyle='--', marker='o')

#pick out the convolutional layers only
conv_layer_idx = [1,2,4,5,7,8,9,11,12,13,15,16,17]
pyplot.plot(np.arange(1,14),
            [rf_per_layer[i] for i in conv_layer_idx],
            linestyle='--', marker='o')

#results match perfectly with araujo paper and Schwartz paper!
#so let's save out 
with open("jsons/theoretical_rf_sizes_VGG16.txt", "w") as fp:
    json.dump(rf_per_layer,fp)

    
    
    
    
    