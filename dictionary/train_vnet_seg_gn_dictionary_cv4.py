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

#numoflandmarks = 20
numofclasses = 3
input_channel = 1
lambda_ce = 1
#lambda_encoding = 0.4
#lambda_structure = 0.5
base_size = 64
#window_str = 8
#stride_str = 4

start_epoch = 0
num_epochs = 30
iterations = 12000
val_iterations = 8277

model = vnet(input_channel, 64, numofclasses, batch_size=1, lambda_ce=lambda_ce)
#model.summary()
#assert 0
#model.load_weights('Unet.best.hd5')

def train_generator(batch_size=1): # train_generator
	# the following parameters need to be assigned values before training
    batch_size = batch_size # very important
    num_classes = 3
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 3 # very important
    patch_stride_regulator = np.array([4, 4, 4]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    data_image_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/images_nii/'
    data_seg_gt_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/label_nii/'
    
    subject_list = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 30, 32, 33, 35, 41, 44, 45, 48, 49, 50, 51, 52, 54, 56, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 202, 203, 205, 206, 216, 218, 219, 220, 221, 225, 226, 227, 231, 236, 238, 242, 243, 246, 247]
    label = 0
    
    while True:
        np.random.shuffle(subject_list)
        for j in range(len(subject_list)):
            subject_index = subject_list[j]

            image = sitk.ReadImage(data_image_path + 'subject{0}.nii.gz'.format(subject_index))
            image = sitk.GetArrayFromImage(image)
            image = image / np.max(image)
            
            segmentation_gt = sitk.ReadImage(data_seg_gt_path + 'subject{0}.nii.gz'.format(subject_index))
            segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
            assert np.all(image.shape == segmentation_gt.shape)
            segmentation_gt = np.transpose(to_categorical(segmentation_gt, num_classes).reshape(list(segmentation_gt.shape + (num_classes,))), [3, 0, 1, 2])

            image_size = image.shape
            seg_gt_size = segmentation_gt.shape
            assert seg_gt_size[1] == image_size[0] and seg_gt_size[2] == image_size[1] and seg_gt_size[3] == image_size[2]
                        
            vertex = np.zeros([200, 3, 6]) ## in the order of (1000, 3, z1, z2, y1, y2, x1, x2)
            shapedata = vertex.shape
            
            patch_index = 0 ## update by 1 after generating a patch
            
            while patch_index < 200:
                # center_z = np.random.randint(0, image_size[0]-int(patch_size[0])) + int(patch_size[0]//2)
                # center_y = np.random.randint(0, image_size[1]-int(patch_size[1])) + int(patch_size[1]//2)
                # center_x = np.random.randint(0, image_size[2]-int(patch_size[2])) + int(patch_size[2]//2)

                center_z = np.random.randint(0, image_size[0])
                center_y = np.random.randint(0, image_size[1])
                center_x = np.random.randint(0, image_size[2])
                
                vertex[patch_index][0] = np.array([center_z-int(patch_size[0]//2), center_z+int(patch_size[0]//2), 
                                                   center_y-int(patch_size[1]//2), center_y+int(patch_size[1]//2), 
                                                   center_x-int(patch_size[2]//2), center_x+int(patch_size[2]//2)])
                
                vertex[patch_index][1] = np.array([center_z-2*int(patch_size[0]//2), center_z+2*int(patch_size[0]//2), 
                                                   center_y-2*int(patch_size[1]//2), center_y+2*int(patch_size[1]//2), 
                                                   center_x-2*int(patch_size[2]//2), center_x+2*int(patch_size[2]//2)])
                
                vertex[patch_index][2] = np.array([center_z-4*int(patch_size[0]//2), center_z+4*int(patch_size[0]//2), 
                                                   center_y-4*int(patch_size[1]//2), center_y+4*int(patch_size[1]//2), 
                                                   center_x-4*int(patch_size[2]//2), center_x+4*int(patch_size[2]//2)])                
                patch_index += 1
    
            modulo=np.mod(shapedata[0], batch_size)
            if modulo!=0:
                num_to_add=batch_size-modulo
                inds_to_add=np.random.randint(0, shapedata[0], num_to_add)
                to_add = vertex[inds_to_add]
                new_vertex = np.vstack((vertex, to_add))
            else:
                new_vertex = vertex.copy()
            
            np.random.shuffle(new_vertex)
            for i_batch in range(int(new_vertex.shape[0]/batch_size)):
                subvertex = new_vertex[i_batch*batch_size:(i_batch+1)*batch_size]
                for count in range(batch_size):
                    ## size_*size_*size_ ##
                    image_one = np.zeros([size_, size_, size_], dtype=np.float32)
                    seg_gt_one = np.zeros([num_classes, size_, size_, size_], dtype=np.float32)
                    seg_gt_one[0] = np.ones([size_, size_, size_], dtype=np.float32) ## I made a huge mistake here ##
                    
                    copy_from, copy_to = corrected_crop(subvertex[count][0], np.array(list(image_size)))

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
                            image[cf_z_lower_bound:cf_z_higher_bound,
                                  cf_y_lower_bound:cf_y_higher_bound,
                                  cf_x_lower_bound:cf_x_higher_bound]

                    seg_gt_one[:,
                            int(copy_to[0]):copy_to[1],
                            int(copy_to[2]):copy_to[3],
                            int(copy_to[4]):copy_to[5]] = \
                            segmentation_gt[:,
                                            cf_z_lower_bound:cf_z_higher_bound,
                                            cf_y_lower_bound:cf_y_higher_bound,
                                            cf_x_lower_bound:cf_x_higher_bound]
                    
                    image_one = np.expand_dims(image_one, axis=0)
                    image_one = np.expand_dims(image_one, axis=0)
                    
                    seg_gt_one = np.expand_dims(seg_gt_one, axis=0)
                    
                    if label == 0:
                        Img = image_one
                        seg_gt = seg_gt_one
                        label += 1
                    else:
                        Img = np.vstack((Img, image_one))
                        seg_gt = np.vstack((seg_gt, seg_gt_one))
                        label += 1
                    
                    if np.remainder(label, batch_size)==0:
                        yield ([Img, seg_gt], [])
                        label = 0

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

def validation_generator(batch_size=1): #validation_generator
	# the following parameters need to be assigned values before training
    batch_size = batch_size # very important
    num_classes = 3
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([1, 1, 1]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    data_image_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/images_nii/'
    data_seg_gt_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/label_nii/'

    subject_list = [253, 255, 257, 258, 260, 262, 263, 264, 265, 266, 267, 269, 270, 275, 283, 284, 286, 287, 288, 289]
    
    while True:
        label = 0
        for idx in range(len(subject_list)):
            subject_index = subject_list[idx]
            
            image = sitk.ReadImage(data_image_path + 'subject{0}.nii.gz'.format(subject_index))
            image = sitk.GetArrayFromImage(image)
            image = image / np.max(image)
            
            segmentation_gt = sitk.ReadImage(data_seg_gt_path + 'subject{0}.nii.gz'.format(subject_index))
            segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
            assert np.all(image.shape == segmentation_gt.shape)
            segmentation_gt = np.transpose(to_categorical(segmentation_gt, num_classes).reshape(list(segmentation_gt.shape + (num_classes,))), [3, 0, 1, 2])
            
            image_size = np.array(np.shape(image), dtype=np.int16)
            seg_gt_size = segmentation_gt.shape
            assert seg_gt_size[1] == image_size[0] and seg_gt_size[2] == image_size[1] and seg_gt_size[3] == image_size[2]
            
            expanded_image_size = (np.ceil(image_size/(1.0*stride))*stride).astype(np.int16)
            
            expanded_image = np.zeros(expanded_image_size, dtype=np.float32)
            expanded_image[0:image_size[0], 0:image_size[1], 0:image_size[2]] = image
        
            expanded_seg_gt = np.zeros([num_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
            expanded_seg_gt[0] = np.ones([expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32) ## I really made a huge mistake here ##
            expanded_seg_gt[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]] = segmentation_gt
            
            num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int16)
            
            total_num_of_patches = np.prod(num_of_patch_with_overlapping)
            
            num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
            num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
            num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
            
            # print("total number of patches in the image is {0}".format(total_num_of_patches))
        
            center = np.zeros([total_num_of_patches, 3]) ## in the order of (total_num_of_patches, 3) ## (384, 3) ##
            
            patch_index = 0
            
            for ii in range(0, num_patch_z):
                for jj in range(0, num_patch_y):
                    for kk in range(0, num_patch_x):
                        center[patch_index] = np.array([int(ii*stride[0] + patch_size[0] // 2),
                                                        int(jj*stride[1] + patch_size[1] // 2),
                                                        int(kk*stride[2] + patch_size[2] // 2)])
                        patch_index += 1
        
            modulo=np.mod(total_num_of_patches, batch_size)
            
            if modulo!=0:
                num_to_add=batch_size-modulo
                inds_to_add=np.random.randint(0, total_num_of_patches, num_to_add) ## the return value is a ndarray
                to_add = center[inds_to_add]
                new_center = np.vstack((center, to_add))
            else:
                new_center = center
            
            #np.random.shuffle(new_center)
            #new_center = new_center[:300, :]

            for i_batch in range(int(new_center.shape[0]/batch_size)):
                subvertex = new_center[i_batch*batch_size:(i_batch+1)*batch_size]
                for count in range(batch_size):
                    ## size_*size_*size_ ##
                    image_one = np.zeros([size_, size_, size_], dtype=np.float32)
                    seg_gt_one = np.zeros([num_classes, size_, size_, size_], dtype=np.float32)
                    seg_gt_one[0] = np.ones([size_, size_, size_], dtype=np.float32) ## To make sure voxels in the padded part are assaigned with label 0 ##
                    
                    z_lower_bound = int(subvertex[count][0] - patch_size[0]//2)
                    z_higher_bound = int(subvertex[count][0] + patch_size[0]//2)
                    y_lower_bound = int(subvertex[count][1] - patch_size[1]//2)
                    y_higher_bound = int(subvertex[count][1] + patch_size[1]//2)
                    x_lower_bound = int(subvertex[count][2] - patch_size[2]//2)
                    x_higher_bound = int(subvertex[count][2] + patch_size[2]//2)
                    
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

                    seg_gt_one[:,
                               int(copy_to[0]):copy_to[1],
                               int(copy_to[2]):copy_to[3],
                               int(copy_to[4]):copy_to[5]] = \
                               expanded_seg_gt[:,
                                               cf_z_lower_bound:cf_z_higher_bound,
                                               cf_y_lower_bound:cf_y_higher_bound,
                                               cf_x_lower_bound:cf_x_higher_bound]
                    
                    image_one = np.expand_dims(image_one, axis=0)
                    image_one = np.expand_dims(image_one, axis=0)
                    
                    seg_gt_one = np.expand_dims(seg_gt_one, axis=0)
                    
                    if label == 0:
                        Img = image_one
                        seg_gt = seg_gt_one
                        label += 1
                    else:
                        Img = np.vstack((Img, image_one))
                        seg_gt = np.vstack((seg_gt, seg_gt_one))
                        label += 1
                    
                    if np.remainder(label, batch_size)==0:
                        yield ([Img, seg_gt], [])
                        label = 0

def lr_schedule(epoch):
    if epoch <= 1:
        lr = 0.0002
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 3:
        lr = 0.0001
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 8:
        lr = 5e-5
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 12:
        lr = 2e-5
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    else:
        lr = 1e-5
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr

def Train():
    train_gen = train_generator(batch_size = 1)
    val_gen = validation_generator(batch_size = 1)
    
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    # start from checkpoints
    if start_epoch > 0:
        model.load_weights('./checkpoints/Vnet.best.hd5_epoch20')
    
    best_full_loss = 1000000000.0
    for epoch in range(start_epoch, num_epochs):
        #K.set_value(model.optimizer.lr, lr_schedule(epoch+1))
        for i_iter in range(iterations):
            # Train with labeled data
            [image1, batch_label], _ = next(train_gen)
            
            loss, loss_ce = model.train_on_batch([image1, batch_label], [])

            if (i_iter+1) % 200 == 0:
                print('Epoch:{0:2d}, iter = {1:5d}, loss = {2:.4f}, loss_ce = {3:.4f}'.format(epoch+1, i_iter, loss, loss_ce))
        
        # Validation
        loss_sum = 0.
        for vi_iter in range(val_iterations):
            [image1, batch_label], _ = next(val_gen)
            val_loss, val_loss_ce = model.test_on_batch([image1, batch_label], [])
            loss_sum += val_loss_ce/val_iterations
        
        current_loss = loss_sum
        print("Validation loss is {0}".format(current_loss))
        if current_loss < best_full_loss:
            best_full_loss = current_loss
            model.save_weights('./checkpoints/Vnet.best.hd5')

if __name__ == '__main__':

    Train()

