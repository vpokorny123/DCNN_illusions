#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:11:32 2023

@author: pokor076
"""
import numpy as np
from matplotlib import pyplot as plt


center_amplitude = 2
max_surround_amp = 5
surround_amplitude = np.arange(0,max_surround_amp,.25)
layers_to_plot = [18]
all_means = []
n = len(layers_to_plot)
colors = plt.cm.viridis(np.linspace(.6,1,n))
x = surround_amplitude-center_amplitude  
y_error_all = []
for i, layer in enumerate(layers_to_plot):
    all_activation = np.load('npys/contrast_contrast_curves_layer%s.npy' %(layer))
    mean_act = np.mean(all_activation,0)
    y_error = []
    boot_means = []
    for j, _ in enumerate(surround_amplitude):
        for _ in range(1000):
            bootsample = np.random.choice(np.transpose(all_activation)[j],
                                          size=np.size(all_activation,0), 
                                          replace=True)
            boot_means.append(bootsample.mean())
        y_error.append(np.std(boot_means)*2)
    y_error_all.append(y_error)
    plt.plot(x, mean_act, color = colors[i])
    plt.errorbar(x, mean_act, yerr = y_error_all[i], color = colors[i])
legend_names = ['Layer '+ str(x) for x in layers_to_plot]
plt.legend(legend_names, framealpha=.1)
plt.xlabel('Surround-Center Contrast Difference')
plt.ylabel('Average Filter Activations')
plt.title('Contrast-Contrast Illusion Responses')
#plt.rcParams['axes.facecolor'] = 'white'
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)
plt.savefig('%s.png' %(layers_to_plot), transparent = True, dpi = 300)

 


    