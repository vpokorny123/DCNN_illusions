#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:23:21 2023

@author: pokor076
"""
import numpy as np
from matplotlib import pyplot
import matplotlib.patches as patches
import tensorflow as tf
from keras.models import Model
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input
from sklearn import preprocessing


def create_grating(size, orientation_rad, spatial_frequency, phase, 
                   radius, amplitude):
    # Create 2D coordinate grid
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    # Compute distance from center of image
    x_center, y_center = size[0] / 2, size[1] / 2
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    # Create circular mask
    mask = np.zeros(size)
    mask[r < radius] = 1
    # Compute grating
    k = 2*np.pi*spatial_frequency
    grating = amplitude * np.sin(k * (np.cos(orientation_rad)*x + np.sin(orientation_rad)*y) + phase)
    # Apply circular mask
    masked_grating = grating * mask
    #vgg16 needs a 3 dimensional RGB image so we'll just concatenate the image to 
    #itself 3 times and shuffle around the dimensions
    img = expand_dims(masked_grating, axis=0)
    img= np.concatenate((img, img, img),axis = 0)
    img = np.moveaxis(img,0,2)
    img = expand_dims(img, axis=0)
    # Display grating
    return img
   

# Define parameters
save_plots = 1
spatial_frequencies = np.round(np.arange(.01,.3,.01),2) # in cycles per pixel
orientation_deg = 45 # in degrees
orientation_rad = np.deg2rad(orientation_deg) # convert to radians
phase = 0 # in radians
amplitude = 1
size = (224, 224) # in pixels
radius = 50 # in pixels
layer = 1
cmap = 'gray'
model = tf.keras.applications.vgg16.VGG16()
short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
short_model.summary()
activation = []
j = 1
for spatial_frequency in spatial_frequencies:
    img = create_grating(size, orientation_rad, spatial_frequency, phase, 
                   radius, amplitude)
    feature_maps = np.squeeze(short_model.predict(img))
    for j in range(64):
        ax = pyplot.subplot(8,8,j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps[:,:,j],cmap = cmap)
    if save_plots == 1: 
        pyplot.savefig('pngs/%s.png' %(spatial_frequency), transparent=True, dpi = 300)
    pyplot.show()
    





