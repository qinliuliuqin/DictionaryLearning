from keras.engine import Layer, InputSpec
from keras import initializers
from keras import backend as K
import tensorflow as tf

class DictionaryLayer(Layer):

    def __init__(self,
                 numChannels,
                 numCodewords,
                 axis=1,
                 codewords_initializer='glorot_uniform',
                 scale_dim_initializer='zeros',
                 scale_center_initializer='zeros',
                 **kwargs):
        super(DictionaryLayer, self).__init__(**kwargs)
        self.axis = axis
        self.numChannels = numChannels
        self.numCodewords = numCodewords
        self.codewords_initializer = initializers.get(codewords_initializer)
        self.scale_dim_initializer = initializers.get(scale_dim_initializer)
        self.scale_center_initializer = initializers.get(scale_center_initializer)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        self.codewords = self.add_weight(shape=(1, self.numCodewords, self.numChannels),
                                     name='codewords',
                                     initializer=self.codewords_initializer)

        self.scale_dim = self.add_weight(shape=(1, 1, 1, self.numChannels,),
                                    name='scale_dim',
                                    initializer=self.scale_dim_initializer)

        self.scale_center = self.add_weight(shape=(1, 1, self.numCodewords,),
                                            name='scale_center',
                                            initializer=self.scale_center_initializer)
        
        self.built = True

    def call(self, inputs, training=None):        
        input_shape = K.int_shape(inputs)

        # Prepare broadcasting shape.
        ndim = len(input_shape)
        assert ndim == 5, 'Only 5D inputs are supported'
        
        if self.axis==1:
            inputs  = tf.transpose(inputs, [0, 2, 3, 4, 1])
            shape = K.int_shape(inputs)
            inputs  = tf.reshape(inputs, [tf.shape(inputs)[0], shape[1]*shape[2]*shape[3], shape[4], 1])
            inputs  = tf.transpose(inputs, [0, 1, 3, 2])
        else:
            shape = K.int_shape(inputs)
            inputs  = tf.reshape(inputs, [tf.shape(inputs)[0], shape[1]*shape[2]*shape[3], shape[4], 1])
            inputs  = tf.transpose(inputs, [0, 1, 3, 2])

        codewords = tf.tile(self.codewords, [tf.shape(inputs)[0], 1, 1])

        # Residual vectors
        R = inputs - tf.expand_dims(codewords, axis=1)
        R_square = tf.square(R)

        weighted_dis = tf.reduce_sum(tf.multiply(R_square, tf.math.exp(self.scale_dim)), axis=-1)
        weight = tf.nn.softmax(-1.0*tf.multiply(weighted_dis, tf.math.exp(self.scale_center)), axis=-1)
        
        recombination = tf.matmul(weight, codewords)
        
        if self.axis == 1:
            recombination = tf.transpose(recombination, [0, 2, 1])
            recombination = tf.reshape(recombination, [tf.shape(recombination)[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]])
        else:
            recombination = tf.reshape(recombination, [tf.shape(recombination)[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]])

        return recombination
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'numChannels': self.numChannels,
            'numCodewords': self.numCodewords,
            'codewords_initializer': initializers.serialize(self.codewords_initializer),
            'scale_dim_initializer': initializers.serialize(self.scale_dim_initializer),
            'scale_center_initializer': initializers.serialize(self.scale_center_initializer),
        }
        base_config = super(DictionaryLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == '__main__':
    from keras.layers import Input

    inputs = Input([32, 8, 8, 8])
    output = DictionaryLayer(32, 16, axis=1)(inputs)
