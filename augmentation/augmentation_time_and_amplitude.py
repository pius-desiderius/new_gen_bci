# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:30:59 2023

@author: Andrei Miroshnikov
"""
import numpy as np
import emd
from scipy.interpolate import CubicSpline

def amplitude_jitter(epochs_data,
                     noise_power=5,
                     fold=1, 
                     depth=1,
                     early_peak=None, 
                     late_peak=None
                    ):
    
    """Function that adds Gaussian noise to the chosen time range in epochs. 
    The function has recursive properties which are defined by depth param.
    If you set 'early_peak' and 'late_peak' to some values (tuple, list or array),
    noise will be added only to the time ranges within these peaks.
    Otherwise, noise will be added to the randomly 
    selected time window withing the entire epoch.
    
    Args:
        epochs_data (numpy.ndarray): 
            A 3d array of epochs with shapes N_epochs, N_channels, N_timestamps
        noise_power (int or float):
            Defines the power of noise added. Higher the value, more noise added.
        fold (int): 
            Defines the magnitude of epochs multiplication. E.g., if fold=5, the function will
            return 5xN_epochs, N_channels, N_timestamps array.
        depth (int): 
            Defines the depth of recursion. If depth !=0, noise will be added recurrently
            to the selected time window (if peaks were provided) or randomly.
            WARNING: If fold and depth are both big enough, the computation can be rough.
            It's better to perform multiplication and recursion separately,
            with fold=N and depth=0 for pure multiplication, 
            and fold=1 and depth=N for pure recursion.
        early_peak (tuple(time_min, time_mean, time_max)):
            Defines the time limits of the early peak. Requires 3 values:
            time_min, time_mean, time_max. Will also work on any ordered iterable.
        late_peak (tuple(time_min, time_mean, time_max)):
            Defines the time limits of the late peak. Requires 3 values: 
            time_min, time_mean, time_max. Will also work on any ordered iterable.
    Returns:
        numpy.ndarray: The 3d array of augemented epochs.

    """
    
    container_augmentated_epochs = []
    
    for _ in range(fold):
        for i in range(epochs_data.shape[0]):
            single_epoch = epochs_data[i, :, :]
            template = np.zeros(shape=single_epoch.shape)
            
            if early_peak and late_peak:
                win1_start = np.random.randint(early_peak[0]//2, early_peak[1])
                win1_end = np.random.randint(early_peak[1], early_peak[2]+early_peak[0]//2)

                win2_start = np.random.randint(late_peak[0]//2, late_peak[1])
                win2_end = np.random.randint(late_peak[1], late_peak[2]+late_peak[0]//2)
            
                noise_arr_1 = np.random.uniform(-np.abs(np.mean(single_epoch)/(1/noise_power)), 
                                                np.abs(np.mean(single_epoch)/(1/noise_power)), 
                                                [win1_end-win1_start,])
                
                noise_arr_2 = np.random.uniform(-np.abs(np.mean(single_epoch)/(1/noise_power)), 
                                                np.abs(np.mean(single_epoch)/(1/noise_power)), 
                                                [win2_end-win2_start,])
                        
                template[:, win1_start:win1_end] = noise_arr_1
                template[:, win2_start:win2_end] = noise_arr_2
                single_epoch = single_epoch + template
                container_augmentated_epochs.append(single_epoch)
    
            else:
                win1 = np.random.randint(0, single_epoch.shape[1])
                win2 = np.random.randint(0, single_epoch.shape[1])
                if win1 > win2:
                    win1, win2 = win2, win1
                    
                noise_arr = np.random.uniform(-np.abs(np.mean(single_epoch*noise_power)), 
                                                np.abs(np.mean(single_epoch*noise_power)), 
                                                [win2-win1,])
                
                template[:, win1:win2] = noise_arr
                single_epoch = single_epoch + template
                container_augmentated_epochs.append(single_epoch)
            
    augmentated_epochs = np.stack(container_augmentated_epochs)
    
    if depth != 0:
        print('RECURSION IS GOING ON!')
        depth = depth - 1
        return amplitude_jitter(augmentated_epochs, 
                                noise_power,
                                fold, 
                                depth, 
                                early_peak, 
                                late_peak
                               )
    else:
        return augmentated_epochs
        



def amplitude_pitcher(epochs_data,
                      noise_power=5, 
                      fold=1, 
                      depth=1, 
                      max_imf=5, 
                      imf_pick=4
                     ):  
    
    """Function that makes shift in epochs amplitudes 
    using Hilbert-Huang transforms of epochs. Requires emd library.
    
    
    Args:
        epochs_data (numpy.ndarray): 
            A 3d array of epochs with shapes N_epochs, N_channels, N_timestamps
        noise_power (int or float):
            Defines the power of noise added. Higher the value, more noise added.
        fold (int): 
            Defines the magnitude of epochs multiplication. E.g., if fold=5, the function will
            return 5xN_epochs, N_channels, N_timestamps array.
        depth (int): 
            Defines the depth of recursion. If depth !=0, noise will be added recurrently
            to the selected time window (if peaks were provided) or randomly.
            WARNING: If fold and depth are both big enough, the computation can be rough.
            It's better to perform multiplication and recursion separately,
            with fold=N and depth=0 for pure multiplication, 
            and fold=1 and depth=N for pure recursion.
        max_imf (int):
            Defines the number of components we extract from epochs.
            Depends on the length of one epoch -- if it is not long enough,
            can result in yielding less components.
        imf_pick (int):
            Defines the component we will add as noise to the epochs.
            Depends on the max_imf -- possible values are [0, max_imf-1].
            If epochs are not long enough to yield the requested amount of components,
            providing too big imf_pick can result in indexing error.
    Returns:
        numpy.ndarray: The 3d array of augemented epochs.

    """
    
    container_augmentated_epochs = []
    
    for _ in range(fold):
        for i in range(epochs_data.shape[0]):
            single_epoch = epochs_data[i, :, :]
            template = np.zeros(shape=single_epoch.shape)
            
            win1 = 0
            win2 = single_epoch.shape[1]-1


            noise_arr = np.random.uniform(-np.abs(np.mean(single_epoch)*noise_power), 
                                            np.abs(np.mean(single_epoch)*noise_power), 
                                            [win2-win1,])
                        
            imf = emd.sift.mask_sift(noise_arr, max_imfs=max_imf, nprocesses=8)
            template[:, win1:win2] = imf[:, imf_pick]
            single_epoch = single_epoch + template
            container_augmentated_epochs.append(single_epoch)
            
    augmentated_epochs = np.stack(container_augmentated_epochs)            
    if depth != 0:
        print('RECURSION IS GOING ON!')
        depth = depth - 1
        return amplitude_pitcher(augmentated_epochs, 
                                 noise_power, 
                                 fold, 
                                 depth, 
                                 max_imf, 
                                 imf_pick)
    else:
        return augmentated_epochs
    
    
def time_warp_spline(epochs_data, time_shift_window=[-15, 15]):
    """Function that warps epochs data in time domain using cubic splines.
    
    Args:
        epochs_data (numpy.ndarray): 
            A 3d array of epochs with shapes N_epochs, N_channels, N_timestamps
        time_window (list):
            A time window of timestamps in which a shift step is defined.
            The step is the same for channels within one epoch but
            vary in epochs; variance is within the time window.
            
    Returns:
        numpy.ndarray: The 3d array of augemented epochs.

    """
    epochs_accumulator = []
    for i in range(epochs_data.shape[0]):
        time_shift = np.random.randint(low=time_shift_window[0], 
                                    high=time_shift_window[1]
                                    )
        single_epoch = epochs_data[i, :, :]
        channels_accumulator = []
        
        for channel_idx in range(single_epoch.shape[0]):
            x = np.arange(single_epoch.shape[1])
            y = single_epoch[channel_idx, :]
            cs = CubicSpline(x, y)
            xs = np.arange(time_shift, len(x)+time_shift, 1)
            shifted_data = cs(xs)
            channels_accumulator.append(shifted_data)
            
        shifted_epoch = np.stack(channels_accumulator, axis=0)
        epochs_accumulator.append(shifted_epoch)
    shifted_data = np.stack(epochs_accumulator, axis=0)
    
    return shifted_data