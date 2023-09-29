#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:20:16 2023

@author: pokor076
"""
import sys
sys.path.append('DCNN_illusions')
import local_functions
import tensorflow as tf
import json
import numpy as np
from keras.models import Model
from matplotlib import pyplot


filter_num = 0
layer = 3
plot_feature_maps = 0
img_size = (224,224)
model_name = 'resnet'

def main():
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
        img = local_functions.create_grating(img_size, ori, spatial_frequency, phase, 
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
            #ax = pyplot.gca()
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
    img = local_functions.create_grating(img_size, ori, spatial_frequency, phase, 
                   max_radius, amplitude)
    feature_maps = np.squeeze(short_model.predict(img))
    
    #save out
    gsf_results = grid_search_results
    gsf_results['Best Radius'] = max_radius
    
    with open("jsons/gsf_results_layer%s_filter%s_%s.txt" 
              %(layer,filter_num,model_name), "w") as fp:
        json.dump(gsf_results,fp)

if __name__ == '__main__':
    main()




    
    
    
    
    