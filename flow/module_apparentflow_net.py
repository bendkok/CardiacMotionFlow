""" The module of ApparentFlow-net """

import sys
sys.path.append('..')

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    UpSampling2D
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D
)
from keras.layers.core import (
    Reshape,
    Lambda
)
from keras.layers.merge import (
    Add,
    Concatenate
)
from keras import backend as K


from helpers import (
    conv_bn_leakyrelu_repetition_block,
    handle_dim_ordering,
    one_hot
)


def net_module(input_shape, num_outputs):
    """Builds a net architecture.
    Args:
        input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels)
        num_outputs: The number of outputs at final softmax layer
        Returns:
            The keras `Model`.
    """
    CHANNEL_AXIS = 3
    handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")

    # Permute dimension order if necessary
    # if K.image_dim_ordering() != 'tf':
    if K.image_data_format() != 'channels_last':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    input_img0 = Input(shape=input_shape, name="input_img0")

    input_img1 = Input(shape=input_shape, name="input_img1")


    concatenate = Concatenate(axis=CHANNEL_AXIS, name="concatenate")([input_img0,
                                                                      input_img1])


    base_channel = 24




    block_conv_1 = conv_bn_leakyrelu_repetition_block(filters=1*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="conv_block1")(concatenate)


    block_pool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', 
        data_format=None, name="pool_block2")(block_conv_1)

    block_conv_2 = conv_bn_leakyrelu_repetition_block(filters=2*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="conv_block2")(block_pool_2)


    block_pool_4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', 
        data_format=None, name="pool_block4")(block_conv_2)

    block_conv_4 = conv_bn_leakyrelu_repetition_block(filters=4*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="conv_block4")(block_pool_4)


    block_pool_8 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', 
        data_format=None, name="pool_block8")(block_conv_4)

    block_conv_8 = conv_bn_leakyrelu_repetition_block(filters=8*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="conv_block8")(block_pool_8)


    block_pool_16 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', 
        data_format=None, name="pool_block16")(block_conv_8)

    block_conv_16 = conv_bn_leakyrelu_repetition_block(filters=16*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="conv_block16")(block_pool_16)





    block_up_8 = UpSampling2D(size=(2,2), name="up_block8")(block_conv_16)

    block_concat_8 = Concatenate(axis=CHANNEL_AXIS, name="concat8")([block_up_8, block_conv_8])    

    block_expan_conv_8 = conv_bn_leakyrelu_repetition_block(filters=8*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="expan_conv_block8")(block_concat_8)


    block_up_4 = UpSampling2D(size=(2,2), name="up_block4")(block_expan_conv_8)

    block_concat_4 = Concatenate(axis=CHANNEL_AXIS, name="concat4")([block_up_4, block_conv_4])    

    block_expan_conv_4 = conv_bn_leakyrelu_repetition_block(filters=4*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="expan_conv_block4")(block_concat_4)


    block_up_2 = UpSampling2D(size=(2,2), name="up_block2")(block_expan_conv_4)

    block_concat_2 = Concatenate(axis=CHANNEL_AXIS, name="concat2")([block_up_2, block_conv_2])    

    block_expan_conv_2 = conv_bn_leakyrelu_repetition_block(filters=2*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="expan_conv_block2")(block_concat_2)


    block_up_1 = UpSampling2D(size=(2,2), name="up_block1")(block_expan_conv_2)

    block_concat_1 = Concatenate(axis=CHANNEL_AXIS, name="concat1")([block_up_1, block_conv_1])    

    block_expan_conv_1 = conv_bn_leakyrelu_repetition_block(filters=1*base_channel, kernel_size=(3,3),     
        repetitions=2, first_layer_down_size=False, alpha=0.0, 
        name="expan_conv_block1")(block_concat_1)






    block_seg_4 = Conv2D(filters=num_outputs, kernel_size=(1,1), strides=(1,1), 
        padding="same", data_format=None, dilation_rate=(1, 1), activation=None,
        use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros",
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None,
        name="seg_block4")(block_expan_conv_4)

    block_seg_2 = Conv2D(filters=num_outputs, kernel_size=(1,1), strides=(1,1), 
        padding="same", data_format=None, dilation_rate=(1, 1), activation=None,
        use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros",
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None,
        name="seg_block2")(block_expan_conv_2)

    block_seg_1 = Conv2D(filters=num_outputs, kernel_size=(1,1), strides=(1,1), 
        padding="same", data_format=None, dilation_rate=(1, 1), activation=None,
        use_bias=True, kernel_initializer="he_normal", bias_initializer="zeros",
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None,
        name="seg_block1")(block_expan_conv_1)

    block_seg_up_2 = UpSampling2D(size=(2,2), name="seg_up_block2")(block_seg_4)

    block_add_2 = Add(name="add_block2")([block_seg_up_2, block_seg_2])

    block_seg_up_1 = UpSampling2D(size=(2,2), name="seg_up_block1")(block_add_2)

    output = Add(name="output")([block_seg_up_1, block_seg_1])

    
    
    model = Model(inputs=[input_img0, input_img1], outputs=output)

    return model








