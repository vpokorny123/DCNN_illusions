#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:23:21 2023

@author: pokor076
"""
import sys
sys.path.append('DCNN_illusions')
import local_functions
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from keras.models import Model
import math
#if running from cla and need to install packages pip install works but conda install doesn't
# User defined parameters
save_plots = 1
spatial_frequencies = np.round(np.arange(.01,.3,.01),2) # in cycles per pixel
orientation_deg = 45 # in degrees
orientation_rad = np.deg2rad(orientation_deg) # convert to radians
phase = 0 # in radians
amplitude = 1
size = (224, 224) # in pixels
radius = 50 # in pixels
layer = 4
cmap = 'gray'
model_name = 'vgg16' #either resnet or vgg16

def main():
    if model_name == 'resnet':
        model = tf.keras.applications.resnet50.ResNet50()
    else:
        model = tf.keras.applications.vgg16.VGG16()
    short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    short_model.summary()
    
    #j = 1
    #create a grating for each spatial frequency
    for spatial_frequency in spatial_frequencies:
        img = local_functions.create_grating(size, orientation_rad, spatial_frequency, phase, 
                   radius, amplitude)
        feature_maps = np.squeeze(short_model.predict(img))
        plot_dims = math.ceil(np.sqrt(feature_maps.shape[2]))
        # loop through each feature map
        for j in range(feature_maps.shape[2]):
            ax = pyplot.subplot(plot_dims,plot_dims,j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(feature_maps[:,:,j],cmap = cmap)
        if save_plots == 1: 
            pyplot.savefig('pngs/%s_%s_layer%s.png' %(spatial_frequency,model_name,layer), transparent=True, dpi = 300)
    pyplot.show()
    
if __name__ == "__main__":
    main()
    





