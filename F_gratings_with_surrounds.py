#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:20:16 2023

@author: pokor076
"""
"""

"""
    
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import numpy as np
from keras.models import Model
from matplotlib import pyplot
import matplotlib.patches as patches

layer = 5
filter_num = 1
plot_feature_maps = 1

# load grid_search_results json file that was created by C_grid_search.py
with open("jsons/gsf_results_layer%s_filter%s.txt" 
          %(layer,filter_num), "r") as fp:
    # Load the dictionary from the file
    gsf_results = json.load(fp)
    
with open("jsons/theoretical_rf_sizes_VGG16.txt" , "r") as fp:
    # Load the dictionary from the file
    rfs = json.load(fp)
     
# Define parameters for center grating
spatial_frequency = gsf_results['Best Spatial Frequency']  # in cycles per pixel
phase = 0  # in radians
amplitude = 1
size = (224, 224)  # in pixels
center_radius = gsf_results['Best Radius']  # in pixels (radius of center grating)
center_orientation_deg = gsf_results['Best Orientation']  # in degrees
center_orientation_rad = np.deg2rad(center_orientation_deg)  # convert to radians
rf = rfs[layer]

model = tf.keras.applications.vgg16.VGG16()
short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
neurons_dim = short_model.output_shape[1:3]
neuron_of_interest = round((neurons_dim[0]-1)/2)


def create_center_surround_grating(size, center_orientation_rad, surround_orientation_rad,
                                   spatial_frequency, phase, center_radius, amplitude, rf):
    
    surround_radius = rf #-center_radius 
                                         
    # Create 2D coordinate grid
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

    # Compute distance from center of image
    x_center, y_center = size[0] / 2, size[1] / 2
    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    
    # Create circular masks for center and surrounding ring
    center_mask = np.zeros(size)
    center_mask[r < center_radius] = 1
    
    surround_mask = np.zeros(size)
    surround_mask[(r >= center_radius) & (r < surround_radius)] = 1
    
    # Compute center grating
    k_center = 2 * np.pi * spatial_frequency
    center_grating = amplitude * np.sin(
        k_center * (np.cos(center_orientation_rad) * x + 
                    np.sin(center_orientation_rad) * y) + phase)
    
    # Compute surrounding grating
    k_surround = 2 * np.pi * spatial_frequency
    surround_grating = amplitude * np.sin(
        k_surround * (np.cos(surround_orientation_rad) * x + np.sin(surround_orientation_rad) * y) + phase
    )
    
    # Apply circular masks to both gratings
    center_grating *= center_mask
    surround_grating *= surround_mask
    
    # Combine center and surrounding gratings
    img = center_grating + surround_grating
    
    return img
    # Highlight single pixel
   # highlight_value = np.max(center_grating) + np.max(surround_grating) + 0.5
   # grating[highlight_pixel[1], highlight_pixel[0]] = highlight_value
    
#img = create_center_surround_grating(size, center_orientation_rad, surround_orientation_rad,
 #                                  spatial_frequency, phase, center_radius, amplitude, rf)
#pyplot.imshow(img, cmap = 'gray')

activation = []
for offset in [-90,0,45,90]: #np.arange(-100,101,2): #[0,180]: #np.arange(0,541):
    surround_orientation_deg = center_orientation_deg + offset  # in degrees (perpendicular to center)
    surround_orientation_rad = np.deg2rad(surround_orientation_deg)  # convert to radians
    img = create_center_surround_grating(size, center_orientation_rad, surround_orientation_rad,
                                       spatial_frequency, phase, center_radius, amplitude, rf)
    pyplot.imshow(img,cmap = 'gray')
    pyplot.savefig('%s.png' %(offset), transparent=True, dpi = 300)
    pyplot.show()
    img = np.expand_dims(img, axis=0)
    img = np.concatenate((img, img, img),axis = 0)
    img = np.moveaxis(img,0,2)
    img = np.expand_dims(img, axis=0)
    feature_maps = np.squeeze(short_model.predict(img))
    #pyplot.imshow(feature_maps[:,:,filter_num],cmap = 'gray')
    #pyplot.show()
    activation.append(feature_maps[neuron_of_interest,neuron_of_interest,filter_num])
pyplot.plot(np.arange(-100,101,2),activation)
plt.title('Layer %s' %(layer))



