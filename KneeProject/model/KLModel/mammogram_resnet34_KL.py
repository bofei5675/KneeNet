import os
import warnings
import h5py
import numpy as np

from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from keras.engine import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file


"""The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
def identity_block(input_tensor, kernel_size, filters, bn_axis, stage, block):
    
    if bn_axis == 3:
        data_format='channels_last'
    else:
        data_format='channels_first'
        
    filters1, filters2 = filters
    conv_name_base = 'layer' + str(stage) + '.' + block 
    bn_name_base = 'layer' + str(stage) + '.' + block 
    
    x = Conv2D(filters1, kernel_size, data_format=data_format, use_bias=False,
               padding='same', name=conv_name_base + '.conv1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, #momentum=0.9, epsilon=1e-05,
                           name=bn_name_base + '.bn1')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2, kernel_size, data_format=data_format, use_bias=False,
               padding='same', name=conv_name_base + '.conv2')(x)
    x = BatchNormalization(axis=bn_axis, #momentum=0.9, epsilon=1e-05,
                           name=bn_name_base + '.bn2')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    
    return x


"""A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
def conv_block(input_tensor, kernel_size, filters, bn_axis, stage, block, strides=(2, 2)):
    
    if bn_axis == 3:
        data_format='channels_last'
    else:
        data_format='channels_first'
        
    filters1, filters2 = filters
    conv_name_base = 'layer' + str(stage) + '.' + block  #'res' + str(stage) + block + '_branch'
    bn_name_base = 'layer' + str(stage) + '.' + block  #'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters1, kernel_size, strides=strides, padding='same', data_format=data_format, use_bias=False,
               name=conv_name_base + '.conv1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, #momentum=0.1, epsilon=1e-05, 
                           name=bn_name_base + '.bn1')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2, kernel_size, padding='same', data_format=data_format, use_bias=False,
               name=conv_name_base + '.conv2')(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-05, 
                           name=bn_name_base + '.bn2')(x)
   
    shortcut = Conv2D(filters2, (1, 1), strides=strides, data_format=data_format, use_bias=False, #padding='valid',
                      name=conv_name_base + '.downsample.0')(input_tensor)
    # have this BN layer  for ImageNet
    shortcut = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-05, 
                                  name=bn_name_base + '.downsample.1')(shortcut)
    
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


'''
    Func:
        Builds Artie's ResNet34 network
    Params:
        input_shape = size of image input shape
        pooling = 'avg' or 'max' pooling after convolutional layers
        bn_axis = 1 or 3
        hidden_layer = 0 or 1 hidden layers after global pooling
        optimizer = 'Adam' or 'SGD'
        learning_rate = learning rate 
    Returns:
        model = ResNet34 architecture
'''
def ResNet34(input_shape=None,
             pooling=None, bn_axis=3, hidden_layer=False, 
             optimizer='ADAM', learning_rate=0.001):
    
    if bn_axis == 3:
        data_format='channels_last'
    else:
        data_format='channels_first'
    
    img_input = Input(shape=input_shape)
    
    x = ZeroPadding2D(padding=(3, 3), data_format=data_format, name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', data_format=data_format, use_bias=False, name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-05, name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', data_format=data_format)(x)
    
    x = conv_block(x, 3, [64, 64], bn_axis, stage=1, block='0', strides=(1, 1))
    #x = identity_block(x, 3, [64, 64], bn_axis, stage=1, block='0')
    x = identity_block(x, 3, [64, 64], bn_axis, stage=1, block='1')
    x = identity_block(x, 3, [64, 64], bn_axis, stage=1, block='2')

    x = conv_block(x, 3, [128, 128], bn_axis, stage=2, block='0')
    x = identity_block(x, 3, [128, 128], bn_axis, stage=2, block='1')
    x = identity_block(x, 3, [128, 128], bn_axis, stage=2, block='2')
    x = identity_block(x, 3, [128, 128], bn_axis, stage=2, block='3')

    x = conv_block(x, 3, [256, 256], bn_axis, stage=3, block='0')
    x = identity_block(x, 3, [256, 256], bn_axis, stage=3, block='1')
    x = identity_block(x, 3, [256, 256], bn_axis, stage=3, block='2')
    x = identity_block(x, 3, [256, 256], bn_axis, stage=3, block='3')
    x = identity_block(x, 3, [256, 256], bn_axis, stage=3, block='4')
    x = identity_block(x, 3, [256, 256], bn_axis, stage=3, block='5')

    x = conv_block(x, 3, [512, 512], bn_axis, stage=4, block='0')
    x = identity_block(x, 3, [512, 512], bn_axis, stage=4, block='1')
    x = identity_block(x, 3, [512, 512], bn_axis, stage=4, block='2')
    
    #x = BatchNormalization(axis=bn_axis, name='bn_last')(x)
    #x = Activation('relu')(x)
    
    #x = AveragePooling2D((7, 7), name='avgpool', data_format=data_format)(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(data_format=data_format, name='global_pooling')(x)
    else:
        x = GlobalMaxPooling2D(data_format=data_format, name='global_pooling')(x)

    if hidden_layer == False:
        #x = Flatten()(x)
        #x = Dense(1000, activation = 'sigmoid', name='fc')(x)
        x = Dense(5, activation = 'softmax', name='fc')(x)

    else:
        x = Dense(256, activation = 'relu', name='hidden1')(x) # , weights = np.array(weights['fully_connected'])))
        x = Dropout(drop_rate)(x)
        #x = Dense(1, activation = 'sigmoid', name='fc')(x)
        x = Dense(5, activation = 'softmax', name='fc')(x)

    #inputs = img_input
    model = Model(img_input, x, name='resnet34')
    
    if optimizer == 'SGD':
        model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], 
                      optimizer = SGD(lr=learning_rate, momentum = 0.9))
    
    elif optimizer == 'ADAM':
        model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], 
                      optimizer = Adam(lr = learning_rate))

    elif optimizer == 'RMSPROP':
        model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], 
                      optimizer = RMSprop(lr = learning_rate))
    model.summary()
    return model


'''
    Def:
        Sets weights for Artie ResNet transfer learning
        BN Layers have the form: [gamma, beta, (running) mean, (running) stdev]
        Since we do not get a mean / stdev to initialize, we use 0 mean and 1 stdev / variance
    Params:
        model = Artie's architecture
        view = 'CC' or 'MLO'
        weights = hdf5 of pre-loaded weights
    Returns:
        model = model with initialized weights
'''
def set_weights(model, view, weights):
    
    # 7x7 convolutional block
    model.get_layer('conv1').set_weights([np.array(weights[view + '_view/conv2d/kernel:0'])])
    model.get_layer('bn1').set_weights([np.array(weights[view + '_view/batch_normalization/gamma:0']),
                                        np.array(weights[view + '_view/batch_normalization/beta:0']), 
                                        np.zeros(np.array(weights[view + '_view/batch_normalization/beta:0']).shape),
                                        np.ones(np.array(weights[view + '_view/batch_normalization/beta:0']).shape)])
    # 1st residual block
    model.get_layer('layer1.0.downsample.0').set_weights([np.array(weights[view + '_view/conv2d_1/kernel:0'])])
    model.get_layer('layer1.0.downsample.1').set_weights([np.array(weights[view + '_view/batch_normalization_1/gamma:0']),
                                                          np.array(weights[view + '_view/batch_normalization_1/beta:0']), 
                                                          np.zeros(np.array(weights[view + '_view/batch_normalization_1/beta:0']).shape),
                                                          np.ones(np.array(weights[view + '_view/batch_normalization_1/beta:0']).shape)])
    model.get_layer('layer1.0.conv1').set_weights([np.array(weights[view + '_view/conv2d_2/kernel:0'])])
    model.get_layer('layer1.0.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_2/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_2/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_2/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_2/beta:0']).shape)])
    model.get_layer('layer1.0.conv2').set_weights([np.array(weights[view + '_view/conv2d_3/kernel:0'])])
    model.get_layer('layer1.0.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_3/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_3/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_3/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_3/beta:0']).shape)])
    
    model.get_layer('layer1.1.conv1').set_weights([np.array(weights[view + '_view/conv2d_4/kernel:0'])])
    model.get_layer('layer1.1.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_4/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_4/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_4/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_4/beta:0']).shape)])
    model.get_layer('layer1.1.conv2').set_weights([np.array(weights[view + '_view/conv2d_5/kernel:0'])])
    model.get_layer('layer1.1.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_5/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_5/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_5/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_5/beta:0']).shape)])
    
    model.get_layer('layer1.2.conv1').set_weights([np.array(weights[view + '_view/conv2d_6/kernel:0'])])
    model.get_layer('layer1.2.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_6/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_6/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_6/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_6/beta:0']).shape)])
    model.get_layer('layer1.2.conv2').set_weights([np.array(weights[view + '_view/conv2d_7/kernel:0'])])
    model.get_layer('layer1.2.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_7/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_7/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_7/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_7/beta:0']).shape)])
    
    # 2nd residual block
    model.get_layer('layer2.0.downsample.0').set_weights([np.array(weights[view + '_view/conv2d_8/kernel:0'])])
    model.get_layer('layer2.0.downsample.1').set_weights([np.array(weights[view + '_view/batch_normalization_8/gamma:0']),
                                                          np.array(weights[view + '_view/batch_normalization_8/beta:0']), 
                                                          np.zeros(np.array(weights[view + '_view/batch_normalization_8/beta:0']).shape),
                                                          np.ones(np.array(weights[view + '_view/batch_normalization_8/beta:0']).shape)])
    model.get_layer('layer2.0.conv1').set_weights([np.array(weights[view + '_view/conv2d_9/kernel:0'])])
    model.get_layer('layer2.0.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_9/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_9/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_9/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_9/beta:0']).shape)])
    model.get_layer('layer2.0.conv2').set_weights([np.array(weights[view + '_view/conv2d_10/kernel:0'])])
    model.get_layer('layer2.0.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_10/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_10/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_10/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_10/beta:0']).shape)])
    model.get_layer('layer2.1.conv1').set_weights([np.array(weights[view + '_view/conv2d_11/kernel:0'])])
    model.get_layer('layer2.1.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_11/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_11/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_11/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_11/beta:0']).shape)])
    model.get_layer('layer2.1.conv2').set_weights([np.array(weights[view + '_view/conv2d_12/kernel:0'])])
    model.get_layer('layer2.1.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_12/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_12/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_12/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_12/beta:0']).shape)])
    model.get_layer('layer2.2.conv1').set_weights([np.array(weights[view + '_view/conv2d_13/kernel:0'])])
    model.get_layer('layer2.2.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_13/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_13/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_13/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_13/beta:0']).shape)])
    model.get_layer('layer2.2.conv2').set_weights([np.array(weights[view + '_view/conv2d_14/kernel:0'])])
    model.get_layer('layer2.2.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_14/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_14/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_14/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_14/beta:0']).shape)])
    
    model.get_layer('layer2.3.conv1').set_weights([np.array(weights[view + '_view/conv2d_15/kernel:0'])])
    model.get_layer('layer2.3.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_15/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_15/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_15/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_15/beta:0']).shape)])
    model.get_layer('layer2.3.conv2').set_weights([np.array(weights[view + '_view/conv2d_16/kernel:0'])])
    model.get_layer('layer2.3.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_16/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_16/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_16/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_16/beta:0']).shape)])
    
    # 3rd residual block
    model.get_layer('layer3.0.downsample.0').set_weights([np.array(weights[view + '_view/conv2d_17/kernel:0'])])
    model.get_layer('layer3.0.downsample.1').set_weights([np.array(weights[view + '_view/batch_normalization_17/gamma:0']),
                                                          np.array(weights[view + '_view/batch_normalization_17/beta:0']), 
                                                          np.zeros(np.array(weights[view + '_view/batch_normalization_17/beta:0']).shape),
                                                          np.ones(np.array(weights[view + '_view/batch_normalization_17/beta:0']).shape)])
    
    model.get_layer('layer3.0.conv1').set_weights([np.array(weights[view + '_view/conv2d_18/kernel:0'])])
    model.get_layer('layer3.0.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_18/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_18/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_18/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_18/beta:0']).shape)])
    model.get_layer('layer3.0.conv2').set_weights([np.array(weights[view + '_view/conv2d_19/kernel:0'])])
    model.get_layer('layer3.0.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_19/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_19/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_19/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_19/beta:0']).shape)])
    
    model.get_layer('layer3.1.conv1').set_weights([np.array(weights[view + '_view/conv2d_20/kernel:0'])])
    model.get_layer('layer3.1.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_20/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_20/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_20/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_20/beta:0']).shape)])
    model.get_layer('layer3.1.conv2').set_weights([np.array(weights[view + '_view/conv2d_21/kernel:0'])])
    model.get_layer('layer3.1.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_21/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_21/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_21/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_21/beta:0']).shape)])
    model.get_layer('layer3.2.conv1').set_weights([np.array(weights[view + '_view/conv2d_22/kernel:0'])])
    model.get_layer('layer3.2.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_22/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_22/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_22/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_22/beta:0']).shape)])
    model.get_layer('layer3.2.conv2').set_weights([np.array(weights[view + '_view/conv2d_23/kernel:0'])])
    model.get_layer('layer3.2.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_23/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_23/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_23/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_23/beta:0']).shape)])
    
    model.get_layer('layer3.3.conv1').set_weights([np.array(weights[view + '_view/conv2d_24/kernel:0'])])
    model.get_layer('layer3.3.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_24/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_24/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_24/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_24/beta:0']).shape)])
    model.get_layer('layer3.3.conv2').set_weights([np.array(weights[view + '_view/conv2d_25/kernel:0'])])
    model.get_layer('layer3.3.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_25/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_25/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_25/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_25/beta:0']).shape)])
    
    model.get_layer('layer3.4.conv1').set_weights([np.array(weights[view + '_view/conv2d_26/kernel:0'])])
    model.get_layer('layer3.4.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_26/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_26/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_26/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_26/beta:0']).shape)])
    model.get_layer('layer3.4.conv2').set_weights([np.array(weights[view + '_view/conv2d_27/kernel:0'])])
    model.get_layer('layer3.4.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_27/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_27/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_27/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_27/beta:0']).shape)])
    
    model.get_layer('layer3.5.conv1').set_weights([np.array(weights[view + '_view/conv2d_28/kernel:0'])])
    model.get_layer('layer3.5.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_28/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_28/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_28/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_28/beta:0']).shape)])
    model.get_layer('layer3.5.conv2').set_weights([np.array(weights[view + '_view/conv2d_29/kernel:0'])])
    model.get_layer('layer3.5.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_29/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_29/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_29/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_29/beta:0']).shape)])
    
    # 4th residual block
    model.get_layer('layer4.0.downsample.0').set_weights([np.array(weights[view + '_view/conv2d_30/kernel:0'])])
    model.get_layer('layer4.0.downsample.1').set_weights([np.array(weights[view + '_view/batch_normalization_30/gamma:0']),
                                                          np.array(weights[view + '_view/batch_normalization_30/beta:0']), 
                                                          np.zeros(np.array(weights[view + '_view/batch_normalization_30/beta:0']).shape),
                                                          np.ones(np.array(weights[view + '_view/batch_normalization_30/beta:0']).shape)])
    
    model.get_layer('layer4.0.conv1').set_weights([np.array(weights[view + '_view/conv2d_31/kernel:0'])])
    model.get_layer('layer4.0.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_31/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_31/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_31/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_31/beta:0']).shape)])
    model.get_layer('layer4.0.conv2').set_weights([np.array(weights[view + '_view/conv2d_32/kernel:0'])])
    model.get_layer('layer4.0.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_32/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_32/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_32/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_32/beta:0']).shape)])
    model.get_layer('layer4.1.conv1').set_weights([np.array(weights[view + '_view/conv2d_33/kernel:0'])])
    model.get_layer('layer4.1.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_33/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_33/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_33/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_33/beta:0']).shape)])
    model.get_layer('layer4.1.conv2').set_weights([np.array(weights[view + '_view/conv2d_34/kernel:0'])])
    model.get_layer('layer4.1.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_34/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_34/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_34/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_34/beta:0']).shape)])
    model.get_layer('layer4.2.conv1').set_weights([np.array(weights[view + '_view/conv2d_35/kernel:0'])])
    model.get_layer('layer4.2.bn1').set_weights([np.array(weights[view + '_view/batch_normalization_35/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_35/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_35/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_35/beta:0']).shape)])
    model.get_layer('layer4.2.conv2').set_weights([np.array(weights[view + '_view/conv2d_36/kernel:0'])])
    model.get_layer('layer4.2.bn2').set_weights([np.array(weights[view + '_view/batch_normalization_36/gamma:0']),
                                                 np.array(weights[view + '_view/batch_normalization_36/beta:0']), 
                                                 np.zeros(np.array(weights[view + '_view/batch_normalization_36/beta:0']).shape),
                                                 np.ones(np.array(weights[view + '_view/batch_normalization_36/beta:0']).shape)])
    return model