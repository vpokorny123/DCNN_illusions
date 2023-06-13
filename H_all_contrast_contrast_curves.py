import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf
from keras.models import Model
import json
import os
layer = 18
skip_if_exist = 1
keep_same_seed = 1 # set to 1 if you want the same scrambling to happen the same way every iteration
plot_stim = 0
plot_feature_map = 0


filter_num= 'all' # set to 'all' if you want to average across features
bootstrap_ci = 1
# Define parameters for center grating
spatial_frequency = .1
phase = 0  # in radians
center_amplitude = 2
max_surround_amp = 5
surround_amplitude = np.arange(0,max_surround_amp,.25)
size = (224, 224)  # in pixels
center_orientation_deg = 0 # in degrees
orientation = np.deg2rad(center_orientation_deg)  # convert to radians
#rf = 100
kernel_size = 6

def gkern(l=5, sig=1.5):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sig))
    #kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def create_center_surround_grating(size, orientation, spatial_frequency, phase, 
                                     center_radius, center_amplitude, surround_amplitude,
                                     rf,kernel_size, seed):
    random.seed(seed)
    np.random.seed(seed)
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
    center_grating = center_amplitude * np.sin(
        k_center * (np.cos(orientation+(np.pi/2)) * y) + 
        np.random.normal(loc = 0, scale = 1, size = size[0])
        )
    #center_grating = np.convolve(center_grating, np.ones(10)/10, mode='same')
    center_grating = [x[random.sample(range(0,size[0]),size[0])] for x in center_grating]
    center_grating = [np.convolve(x, gkern(l=kernel_size), mode='same') for x in center_grating]
    center_grating = [np.convolve(x, gkern(l=kernel_size), mode='same') for x in np.transpose(center_grating)]
    
    # Compute surrounding grating
    k_surround = 2 * np.pi * spatial_frequency
    surround_grating = surround_amplitude * np.sin(
        k_surround * (np.cos(orientation) * x + np.sin(orientation) * y) + 
        np.random.normal(loc = 0, scale = 1, size = size[0])
    )
    surround_grating = [x[random.sample(range(0,size[0]),size[0])] for x in surround_grating]
    surround_grating = [np.convolve(x, gkern(l=kernel_size), mode='same') for x in surround_grating]
    surround_grating = [np.convolve(x, gkern(l=kernel_size), mode='same') for x in np.transpose(surround_grating)]

    #surround_grating = [np.convolve(x, np.ones(10)/10, mode='same') for x in surround_grating]
    # Apply circular masks to both gratings
    center_grating *= center_mask
    surround_grating *= surround_mask
    
    # Combine center and surrounding gratings
    img = center_grating + surround_grating
    return img

with open("jsons/theoretical_rf_sizes_VGG16.txt" , "r") as fp:
    # Load the dictionary from the file
    rfs = json.load(fp)
    
    
all_gsf_results = np.load("npys/contrast_gsf_results_layer%s_filterall.npy" %(layer))    
# load grid_search_results json file that was created by C_grid_search.py
#if filter_num == 'all':
#    with open("jsons/contrast_gsf_results_layer%s_filterall.txt" 
#          %(layer), "r") as fp:
#    # Load the dictionary from the file
#        all_gsf_results = json.load(fp)
    

rf = rfs[layer]
model = tf.keras.applications.vgg16.VGG16()
short_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
neurons_dim = short_model.output_shape[1:3]
neuron_of_interest = round((neurons_dim[0]-1)/2)
#if file doesn't exist OR we set script to overwrite via skip_if_exist = 0 then run
#otherwise don't
if not os.path.isfile("npys/contrast_contrast_curves_layer%s.npy" %(layer)) or skip_if_exist == 0: 
    all_activation = []
    for idx, center_radius in enumerate(all_gsf_results):
        if center_radius > 0:
            each_activation = []
            y_error = []
            for delta in surround_amplitude: #np.arange(-100,101,2): #[0,180]: #np.arange(0,541):
                if keep_same_seed ==1:
                    seed = 1 
                img = create_center_surround_grating(size, orientation, spatial_frequency, phase, 
                                                 center_radius, center_amplitude, delta,
                                                 rf, kernel_size, seed)
                if plot_stim == 1:
                    plt.imshow(img,cmap = 'gray')
                    plt.clim(-4,4)
                    plt.show()
                img = np.expand_dims(img, axis=0)
                img = np.concatenate((img, img, img),axis = 0)
                img = np.moveaxis(img,0,2)
                img = np.expand_dims(img, axis=0)
                feature_maps = np.squeeze(short_model.predict(img))
                each_activation.append(feature_maps[neuron_of_interest,neuron_of_interest,idx])
            all_activation.append(each_activation)
            print('done with neuron '+str(idx+1)+' with preferred radius '+str(center_radius) )
    
            
           
            #if plot_feature_map ==1:
            #    plt.imshow(feature_maps[:,:,filter_num],cmap = 'gray')
            #    plt.show()
            #if filter_num == 'all':  
            #    all_neurons = feature_maps[neuron_of_interest,neuron_of_interest,:]
            #    activation.append(np.mean(all_neurons))
            #    if bootstrap_ci == 1:
            #        boot_means = []
            #        for _ in range(10000):
            #            bootsample = np.random.choice(all_neurons,size=512, replace=True)
            #            boot_means.append(bootsample.mean())
            #        y_error.append(np.std(boot_means)*2)
            #    else:
            #        y_error.append(np.std(all_neurons)/np.sqrt(np.size(all_neurons))*2)
            #else:
    np.save('npys/contrast_contrast_curves_layer%s' %(layer), all_activation) 
                # I love np.save! It's easy to write/read, very quick and low storage cost,
                #only issue is portability (e.g. bringing over to R or matlab) in which case
                #csv is probably better option
all_activation = np.load('npys/contrast_contrast_curves_layer%s.npy' %(layer))
x = surround_amplitude-center_amplitude
y = np.mean(all_activation,0)
plt.plot(x,y)
plt.title('Layer %s' %(layer))
plt.xlabel('Surround-Center Contrast Difference')
plt.ylabel('Activation')

