#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:56:08 2023

@author: pokor076
"""
import numpy as np
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
    img = np.expand_dims(masked_grating, axis=0)
    img= np.concatenate((img, img, img),axis = 0)
    img = np.moveaxis(img,0,2)
    img = np.expand_dims(img, axis=0)
    # Display grating
    return img
   