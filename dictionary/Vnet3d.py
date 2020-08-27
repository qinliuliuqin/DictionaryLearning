from GroupNorm3D import GroupNormalization
from Dictionary import DictionaryLayer

import warnings
import numpy as np
from keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Activation, Add, Concatenate, Lambda, Dense, Reshape
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

def custom_categorical_crossentropy(gt, pred):
    gt = tf.cast(gt, tf.float32)
    # manual computation of crossentropy
    epsilon = 1e-6
    pred = tf.clip_by_value(pred, epsilon, 1. - epsilon)
    return - tf.reduce_mean(tf.reduce_sum(gt * tf.log(pred), axis=1), name='crossentropy')

def hybrid_loss(gt, pred):
    return custom_categorical_crossentropy(gt, pred)

def semantic_loss(gt, semantic):
    loss = K.mean(K.binary_crossentropy(gt, semantic, from_logits=False))
    return loss

def vnet(num_input_channel, base_size, numofclasses, batch_size=1, lambda_ce=1., data_format='channels_first'):

    # Layer 1
    if data_format == 'channels_first':
        inputs = Input([num_input_channel,] + [base_size, base_size, base_size])
    else:
        inputs = Input([base_size, base_size, base_size] + [num_input_channel,])
    
    conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=16, axis=1)(conv1)
    conv1 = PReLU()(conv1)

    identity1 = Conv3D(16, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(inputs)
    identity1 = GroupNormalization(groups=16, axis=1)(identity1)
    identity1 = PReLU()(identity1)

    conv1 = Add()([conv1, identity1])

    down1 = Conv3D(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv1)
    down1 = PReLU()(down1)

    conv2 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down1)
    conv2 = GroupNormalization(groups=32, axis=1)(conv2)
    conv2 = PReLU()(conv2)

    conv2 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=32, axis=1)(conv2)
    conv2 = PReLU()(conv2)

    identity2 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down1)
    identity2 = GroupNormalization(groups=32, axis=1)(identity2)
    identity2 = PReLU()(identity2)

    conv2 = Add()([conv2, identity2])

    down2 = Conv3D(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv2)
    down2 = PReLU()(down2)

    conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down2)
    conv3 = GroupNormalization(groups=64, axis=1)(conv3)
    conv3 = PReLU()(conv3)

    conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=64, axis=1)(conv3)
    conv3 = PReLU()(conv3)

    conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=64, axis=1)(conv3)
    conv3 = PReLU()(conv3)
    
    identity3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down2)
    identity3 = GroupNormalization(groups=64, axis=1)(identity3)
    identity3 = PReLU()(identity3)

    conv3 = Add()([conv3, identity3])

    down3 = Conv3D(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    down3 = PReLU()(down3)

    conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down3)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = PReLU()(conv4)

    conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = PReLU()(conv4)

    conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = PReLU()(conv4)
    
    identity4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down3)
    identity4 = GroupNormalization(groups=128, axis=1)(identity4)
    identity4 = PReLU()(identity4)

    conv4 = Add()([conv4, identity4])

    down4 = Conv3D(256, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    down4 = PReLU()(down4)

    conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down4)
    conv5 = GroupNormalization(groups=256, axis=1)(conv5)
    conv5 = PReLU()(conv5)

    conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=256, axis=1)(conv5)
    conv5 = PReLU()(conv5)

    conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=256, axis=1)(conv5)
    conv5 = PReLU()(conv5)
    
    identity5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down4)
    identity5 = GroupNormalization(groups=256, axis=1)(identity5)
    identity5 = PReLU()(identity5)

    conv5 = Add()([conv5, identity5])

    up1 = Conv3DTranspose(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    concat1 = Concatenate(axis=1)([up1, conv4])

    conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat1)
    conv6 = GroupNormalization(groups=256, axis=1)(conv6)
    conv6 = PReLU()(conv6)

    conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=256, axis=1)(conv6)
    conv6 = PReLU()(conv6)

    conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=256, axis=1)(conv6)
    conv6 = PReLU()(conv6)
    
    identity6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up1)
    identity6 = GroupNormalization(groups=256, axis=1)(identity6)
    identity6 = PReLU()(identity6)
    
    conv6 = Add()([conv6, identity6])

    up2 = Conv3DTranspose(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    concat2 = Concatenate(axis=1)([up2, conv3])

    conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat2)
    conv7 = GroupNormalization(groups=128, axis=1)(conv7)
    conv7 = PReLU()(conv7)

    conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=128, axis=1)(conv7)
    conv7 = PReLU()(conv7)

    conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=128, axis=1)(conv7)
    conv7 = PReLU()(conv7)
    
    identity7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up2)
    identity7 = GroupNormalization(groups=128, axis=1)(identity7)
    identity7 = PReLU()(identity7)
    
    conv7 = Add()([conv7, identity7])

    up3 = Conv3DTranspose(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    concat3 = Concatenate(axis=1)([up3, conv2])

    conv8 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat3)
    conv8 = GroupNormalization(groups=64, axis=1)(conv8)
    conv8 = PReLU()(conv8)

    conv8 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv8)
    conv8 = GroupNormalization(groups=64, axis=1)(conv8)
    conv8 = PReLU()(conv8)

    identity8 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up3)
    identity8 = GroupNormalization(groups=64, axis=1)(identity8)
    identity8 = PReLU()(identity8)
    
    conv8 = Add()([conv8, identity8])

    up4 = Conv3DTranspose(16, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv8)
    concat4 = Concatenate(axis=1)([up4, conv1])

    conv9 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat4)
    conv9 = GroupNormalization(groups=32, axis=1)(conv9)
    conv9 = PReLU()(conv9)
    
    identity9 = Conv3D(32, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up4)
    identity9 = GroupNormalization(groups=32, axis=1)(identity9)
    identity9 = PReLU()(identity9)
    
    conv9 = Add()([conv9, identity9])

    conv10 = DictionaryLayer(32, 32, axis=1)(conv9)

    logits = Conv3D(numofclasses, kernel_size=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv10)
    output1 = Lambda(lambda x: K.softmax(x, axis=1))(logits)

    gt = Input((numofclasses, base_size, base_size, base_size))

    seg_loss = Lambda(lambda x: hybrid_loss(*x), name="ce_loss")([gt, output1])
    model = Model(inputs=[inputs, gt], outputs=[output1, seg_loss])
    model.add_loss(lambda_ce * seg_loss)
    
    model.compile(optimizer=Adam(lr=0.0001), loss=[None] * len(model.outputs))

    metrics_names = ["ce_loss"]
    loss_weights = {
        "ce_loss": lambda_ce,
    }
    
    for name in metrics_names:
        layer = model.get_layer(name)
        loss = (layer.output * loss_weights.get(name, 1.))
        model.metrics_tensors.append(loss)
    
    return model

if __name__ == '__main__':
    model = vnet(1, 64, 3)
    model.summary()














