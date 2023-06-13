#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:20:16 2023

@author: pokor076
"""
filter_num = 0
layer = 9
plot_feature_maps = 0

import tensorflow as tf
import json
import numpy as np
from numpy import expand_dims
from keras.models import Model
from matplotlib import pyplot

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

img_size = (224,224)

# load grid_search_results json file that was created by C_grid_search.py
with open("jsons/grid_search_results_layer%s_filter%s.txt" 
          %(layer,filter_num), "r") as fp:
    # Load the dictionary from the file
    grid_search_results = json.load(fp)
    
ori = grid_search_results['Best Orientation']
spatial_frequency = grid_search_results['Best Spatial Frequency']
phase = 0
amplitude = 1
radii = np.arange(2,img_size[0]/2,1)

#load VGG16
model = tf.keras.applications.vgg16.VGG16()
short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
neurons_dim = short_model.output_shape[1:3]
neuron_of_interest = round((neurons_dim[0]-1)/2)

# now we'll look at activation as a function of radius
activation = []
for radius in radii:
    img = create_grating(img_size, ori, spatial_frequency, phase, 
                   radius, amplitude)
    feature_maps = np.squeeze(short_model.predict(img))
    indiv_activation = feature_maps[neuron_of_interest,
                                       neuron_of_interest,
                                       filter_num]
    # Record the activation of the target neuron
    activation.append(indiv_activation)
    if plot_feature_maps==1:
        unscaled_fmap = feature_maps[:,:,filter_num]
        scaled_fmap = (unscaled_fmap - np.min(unscaled_fmap)) / (np.max(unscaled_fmap)
        - np.min(unscaled_fmap))
        pyplot.imshow(scaled_fmap,cmap = 'gray')
        #rect = patches.Rectangle((neuron_of_interest, neuron_of_interest), 1, 1, linewidth=2,
        #                  edgecolor='r', facecolor='r')
        ax = pyplot.gca()
        #ax.add_patch(rect)
        pyplot.clim(0,1)
        pyplot.show()
scaled_activation = (activation- np.min(activation)) / (np.max(activation)
- np.min(activation))
pyplot.plot(radii,activation)
pyplot.show()

pyplot.plot(radii,scaled_activation)
pyplot.show()

#save out max activation radius
max_radius = radii[np.argmax(activation)]

#let's look at the size of best activation
img = create_grating(img_size, ori, spatial_frequency, phase, 
               max_radius, amplitude)
feature_maps = np.squeeze(short_model.predict(img))

#save out
gsf_results = grid_search_results
gsf_results['Best Radius'] = max_radius

with open("jsons/gsf_results_layer%s_filter%s.txt" 
          %(layer,filter_num), "w") as fp:
    json.dump(gsf_results,fp)






    
    
    
    
    