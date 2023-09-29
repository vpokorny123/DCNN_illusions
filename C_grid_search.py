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
import matplotlib.patches as patches
import tensorflow as tf
from keras.models import Model
import json

# Define parameters
layer = 3
filter_num = 0
size = (224, 224) # in pixels
radius = 50 # in pixels
spatial_frequencies = np.round(np.arange(.01,.3,.02),2) # in cycles per pixel
orientation_degs = np.arange(1,180,2) # in degrees
orientation_rads = np.deg2rad(orientation_degs) # convert to radians
phase = 0 # in radians
amplitude = 1
cmap = 'gray'
model_name = 'resnet'

def main():
    if model_name == 'resnet':
        model = tf.keras.applications.resnet50.ResNet50()
    else:
        model = tf.keras.applications.vgg16.VGG16()    
    short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    neurons_dim = short_model.output_shape[1:3]
    neuron_of_interest = round((neurons_dim[0]-1)/2)
    # Loop through each orientation and spatial frequency and record the activation of the target neuron
    best_activation = -np.inf
    best_orientation = None
    best_spatial_frequency = None
    activation = 0 # just pre-allocating
    for ori in orientation_rads:
        for spatial_frequency in spatial_frequencies:
            img = local_functions.create_grating(size, ori, spatial_frequency, phase, 
                           radius, amplitude)
            feature_maps = np.squeeze(short_model.predict(img))
            # Record the activation of the target neuron
            activation = feature_maps[neuron_of_interest,neuron_of_interest,filter_num]
            # Update the best orientation and spatial frequency if the current activation is higher
            if activation > best_activation:
                best_activation = activation
                best_orientation = ori
                best_orientation_deg = np.rad2deg(ori)
                best_spatial_frequency = spatial_frequency
    
    # Print the best orientation and spatial frequency
    print(f"Best orientation: {best_orientation}")
    print(f"Best orientation (degrees): {best_orientation_deg}")
    print(f"Best spatial frequency: {best_spatial_frequency}")
    print(f"Best activation: {best_activation}")
    
    #create dictionary and save out
    grid_search_results = {
        "Best Orientation": np.float64(best_orientation),
        "Best Orientation (Degrees)": np.float64(best_orientation_deg),
        "Best Spatial Frequency": np.float64(best_spatial_frequency),
        "Best Activation": np.float64(best_activation)
        }
    
    with open("jsons/grid_search_results_layer%s_filter%s_%s.txt" 
              %(layer,filter_num,model_name), "w") as fp:
        json.dump(grid_search_results,fp)
    
    
    # now let's see what "best" parameters looks like
    img = local_functions.create_grating(size, best_orientation, best_spatial_frequency, phase, 
                   radius, amplitude)
    feature_maps = np.squeeze(short_model.predict(img))
    # Record the activation of the target neuron
    pyplot.imshow(feature_maps[:,:,filter_num],cmap = 'gray')
    rect = patches.Rectangle((neuron_of_interest, neuron_of_interest), 1, 1, linewidth=2,
                              edgecolor='r', facecolor='r')
    ax = pyplot.gca()
    ax.add_patch(rect)
    pyplot.show()

if __name__ == '__main__':
    main()
