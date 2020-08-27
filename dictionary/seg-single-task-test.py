#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:30:26 2018

@author: xiaoyangchen
"""

from __future__ import print_function
import os
import h5py
import time
import random
import warnings
import numpy as np
import SimpleITK as sitk
from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate, Add
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import backend as K
from keras.utils import to_categorical

import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

from Vnet3d import vnet

def corrected_crop(array, image_size):
    array_ = array.copy()
    image_size_ = image_size.copy()
    
    copy_from = [0, 0, 0, 0, 0, 0] #np.zeros([6,], dtype=np.int32)
    copy_to = [0, 0, 0, 0, 0, 0] #np.zeros([6,], dtype=np.int32)
    ## 0 ##
    if array[0] < 0:
        copy_from[0] = 0
        copy_to[0] = int(abs(array_[0]))
    else:
        copy_from[0] = int(array_[0])
        copy_to[0] = 0
    ## 1 ##
    if array[1] > image_size_[0]:
        copy_from[1] = None
        copy_to[1] = -int(array_[1] - image_size_[0])
    else:
        copy_from[1] = int(array_[1])
        copy_to[1] = None
    ## 2 ##
    if array[2] < 0:
        copy_from[2] = 0
        copy_to[2] = int(abs(array_[2]))
    else:
        copy_from[2] = int(array_[2])
        copy_to[2] = 0
    ## 3 ##
    if array[3] > image_size_[1]:
        copy_from[3] = None
        copy_to[3] = -int(array_[3] - image_size_[1])
    else:
        copy_from[3] = int(array_[3])
        copy_to[3] = None
    ## 4 ##
    if array[4] < 0:
        copy_from[4] = 0
        copy_to[4] = int(abs(array_[4]))
    else:
        copy_from[4] = int(array_[4])
        copy_to[4] = 0
    ## 5 ##  
    if array[5] > image_size_[2]:
        copy_from[5] = None
        copy_to[5] = -int(array_[5] - image_size_[2])
    else:
        copy_from[5] = int(array_[5])
        copy_to[5] = None

    return copy_from, copy_to

def test(batch_size=1):
    # the following parameters need to be assigned values before training
    batch_size = batch_size # very important
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([2, 2, 2]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    save_path = './results/'

    test_data_image_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/images_nii/'

    test_subject_list = [253, 255, 257, 258, 260, 262, 263, 264, 265, 266, 267, 269, 270, 275, 283, 284, 286, 287, 288, 289]
    model = vnet(input_channel, 64, numofclasses, batch_size=1, lambda_ce=1)
    model.load_weights('./checkpoints/Vnet.best.hd5')
    
    for j in range(len(test_subject_list)): ## should be changed to 2 ##
        subject_index = test_subject_list[j]
        
        image = sitk.ReadImage(test_data_image_path + 'subject{0}.nii.gz'.format(subject_index))
        image = sitk.GetArrayFromImage(image)
        image = image/np.max(image)
        
        image_size = np.array(np.shape(image))
        
        expanded_image_size = (np.ceil(image_size/(1.0*stride))*stride).astype(np.int)
        
        expanded_image = np.zeros(expanded_image_size)
        expanded_image[0:image_size[0], 0:image_size[1], 0:image_size[2]] = image
        
        #expanded_image_mean = np.mean(expanded_image)
        #expanded_image_std = np.std(expanded_image)
        #expanded_image = (expanded_image - expanded_image_mean)/expanded_image_std

        predicted_seg = np.zeros([numofclasses, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        # predicted_heatmaps = np.zeros([numoflmks, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)

        count_matrix_seg = np.zeros([numofclasses, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
        # count_matrix_lmk = np.zeros([numoflmks, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)

        num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int16)
        
        total_num_of_patches = np.prod(num_of_patch_with_overlapping)
        
        num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
        num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
        num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
        
        print("total number of patches in the image is {0}".format(total_num_of_patches))
    
        center = np.zeros([total_num_of_patches, 3]) ## in the order of (total_num_of_patches, 3) ## (384, 3) ##
        
        patch_index = 0
        
        print(num_patch_z, num_patch_y, num_patch_x)
        
        for ii in range(0, num_patch_z):
            for jj in range(0, num_patch_y):
                for kk in range(0, num_patch_x):
                    center[patch_index] = np.array([int((ii + patch_stride_regulator[0]//2)*stride[0]),
                                                    int((jj + patch_stride_regulator[1]//2)*stride[1]),
                                                    int((kk + patch_stride_regulator[2]//2)*stride[2])])
                    patch_index += 1
        

        for idx in range(total_num_of_patches):
            ## 96*96*96 ##
            image_one = np.zeros([size_, size_, size_], dtype=np.float32)
            
            z_lower_bound = int(center[idx][0] - patch_size[0]//2)
            z_higher_bound = int(center[idx][0] + patch_size[0]//2)
            y_lower_bound = int(center[idx][1] - patch_size[1]//2)
            y_higher_bound = int(center[idx][1] + patch_size[1]//2)
            x_lower_bound = int(center[idx][2] - patch_size[2]//2)
            x_higher_bound = int(center[idx][2] + patch_size[2]//2)
            
            virgin_range = np.array([z_lower_bound, z_higher_bound, y_lower_bound, y_higher_bound, x_lower_bound, x_higher_bound])
            copy_from, copy_to = corrected_crop(virgin_range, expanded_image_size)
            
            cf_z_lower_bound = int(copy_from[0])
            if copy_from[1] is not None:
                cf_z_higher_bound = int(copy_from[1])
            else:
                cf_z_higher_bound = None
            
            cf_y_lower_bound = int(copy_from[2])
            if copy_from[3] is not None:
                cf_y_higher_bound = int(copy_from[3])
            else:
                cf_y_higher_bound = None
            
            cf_x_lower_bound = int(copy_from[4])
            if copy_from[5] is not None:
                cf_x_higher_bound = int(copy_from[5])
            else:
                cf_x_higher_bound = None
            
            image_one[int(copy_to[0]):copy_to[1],
                      int(copy_to[2]):copy_to[3],
                      int(copy_to[4]):copy_to[5]] = \
                      expanded_image[cf_z_lower_bound:cf_z_higher_bound,
                                     cf_y_lower_bound:cf_y_higher_bound,
                                     cf_x_lower_bound:cf_x_higher_bound]
            
            image_one = np.expand_dims(image_one, axis=0)

            image_1 = np.expand_dims(image_one, axis=0)

            fake_gt = np.zeros([batchsize, numofclasses, size_, size_, size_])
            #fake_class_gt = np.zeros([batchsize, numofclasses])
            ## output batch ##
            predicted_one, _ = model.predict([image_1, fake_gt])
            print(predicted_one.shape)
            
            predicted_seg[:, np.int16(center[idx][0] - patch_size[0]//2):np.int16(center[idx][0] + patch_size[0]//2),
                            np.int16(center[idx][1] - patch_size[1]//2):np.int16(center[idx][1] + patch_size[1]//2),
                            np.int16(center[idx][2] - patch_size[2]//2):np.int16(center[idx][2] + patch_size[2]//2)] += predicted_one[0]

            count_matrix_seg[:, np.int16(center[idx][0] - patch_size[0]//2):np.int16(center[idx][0] + patch_size[0]//2),
                            np.int16(center[idx][1] - patch_size[1]//2):np.int16(center[idx][1] + patch_size[1]//2),
                            np.int16(center[idx][2] - patch_size[2]//2):np.int16(center[idx][2] + patch_size[2]//2)] += 1.0

        predicted_seg_ = predicted_seg/(1.0*count_matrix_seg)
        #predicted_heatmaps_ = predicted_heatmaps/(1.0*count_matrix_lmk)
                        
        output_seg = predicted_seg_[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]]
        #output_heatmaps = predicted_heatmaps_[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]]

        output_label = np.argmax(output_seg, axis=0)
        output_image_to_save = sitk.GetImageFromArray(output_label.astype(np.float32))

        if not os.path.exists(save_path + 'segmentation/subject{0}'.format(subject_index)):
            os.makedirs(save_path + 'segmentation/subject{0}'.format(subject_index))

        sitk.WriteImage(output_image_to_save, save_path + 'segmentation/' + 'subject{0}/multiscale_subject_{0}.nii.gz'.format(subject_index))

batchsize =1
numofclasses = 3
input_channel = 1
numoflmks = 18
#lambda_seg = 1
#lambda_sem = 1
#window_str = 8
#stride_str = 4

test(batch_size=batchsize)

