#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:20:16 2023

@author: pokor076
"""
filter_num = 'all'
layer = 17
plot_feature_maps = 0
plot_stim = 0
scramble = 1

import tensorflow as tf
import json
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
import random

def gkern(l=5, sig=1.5):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sig))
    #kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def create_center_grating(size, orientation, spatial_frequency, phase, 
                                     center_radius, center_amplitude, kernel_size,
                                     scramble):
    np.random.seed(1)
    random.seed(1)
    # Create 2D coordinate grid
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

    # Compute distance from center of image
    x_center, y_center = size[0] / 2, size[1] / 2
    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    
    # Create circular masks for center and surrounding ring
    center_mask = np.zeros(size)
    center_mask[r < center_radius] = 1
    
    # Compute center grating
    k_center = 2 * np.pi * spatial_frequency
    random.seed(10)
    if scramble == 1:
        center_grating = center_amplitude * np.sin(
            k_center * (np.cos(orientation+(np.pi/2)) * y) + 
            np.random.normal(loc = 0, scale = 1, size = size[0])
            )
        #center_grating = np.convolve(center_grating, np.ones(10)/10, mode='same')
        random.seed(10)
        center_grating = [x[random.sample(range(0,size[0]),size[0])] for x in center_grating]
        center_grating = [np.convolve(x, gkern(l=kernel_size), mode='same') for x in center_grating]
        center_grating = [np.convolve(x, gkern(l=kernel_size), mode='same') for x in np.transpose(center_grating)]
    else:
        k = 2*np.pi*spatial_frequency
        center_grating = center_amplitude * np.sin(k * (np.cos(orientation)*x + np.sin(orientation)*y))
    # Apply circular masks to both gratings
    center_grating *= center_mask
    
    # Combine center and surrounding gratings
    img = center_grating 
    return img


size = (224,224)
center_radii = np.arange(0,60,2) # let's shorten to 60 to save time
# Define parameters for center grating
spatial_frequency = .1
phase = 0  # in radians
center_amplitude = 2
center_orientation_deg = 0 # in degrees
orientation = np.deg2rad(center_orientation_deg)  # convert to radians
kernel_size = 6

model = tf.keras.applications.vgg16.VGG16()
short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
neurons_dim = short_model.output_shape[1:3]
neuron_of_interest = round((neurons_dim[0]-1)/2)

if filter_num =='all':
    filters_num = np.arange(0,short_model.output_shape[3:4][0])

# now we'll look at activation as a function of radius
max_radius = []
for filt in filters_num:
    activation = []
    for center_radius in center_radii:
        img = create_center_grating(size, orientation, spatial_frequency, phase, 
                                         center_radius, center_amplitude, kernel_size,
                                         scramble)
        if plot_stim == 1:
            plt.imshow(img)
            plt.show()
        img = np.expand_dims(img, axis=0)
        img = np.concatenate((img, img, img),axis = 0)
        img = np.moveaxis(img,0,2)
        img = np.expand_dims(img, axis=0)
        feature_maps = np.squeeze(short_model.predict(img))
        indiv_activation = feature_maps[neuron_of_interest,
                                       neuron_of_interest,
                                       filt]
    # Record the activation of the target neuron
        activation.append(indiv_activation)
        if plot_feature_maps==1:
            unscaled_fmap = feature_maps[:,:,filt]
            scaled_fmap = (unscaled_fmap - np.min(unscaled_fmap)) / (np.max(unscaled_fmap)
            - np.min(unscaled_fmap))
            plt.imshow(scaled_fmap,cmap = 'gray')
            ax = plt.gca()
            #ax.add_patch(rect)
            plt.clim(0,1)
            plt.show()
    plt.plot(center_radii,activation)
    plt.show()

    #save out max activation radius
    max_radius.append(center_radii[np.argmax(activation)])
    print("done with feature map #%s" %(filt) )

    #save out
contrast_gsf = max_radius

np.save("npys/contrast_gsf_results_layer%s_filter%s"  %(layer,filter_num),
        contrast_gsf)
#with open("jsons/contrast_gsf_results_layer%s_filter%s.txt" 
#          %(layer,filter_num), "w") as fp:
#    json.dump(np.float64(contrast_gsf),fp)






    
    
    
    
    