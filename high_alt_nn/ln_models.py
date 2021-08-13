# ---------------------------------------------#
# model file for lunar net
# Author: Eric Ham
# Last Edited: 7/14/20
# ---------------------------------------------#

from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
from keras import backend as K
from tensorflow.keras.applications import mobilenet_v2
from keras.models import Sequential

# 2D conv regressor network. 

# width is first image dimension (w,l) or (y,x)
# default strides = 1, default filter size --> 3x3
def lunar_net_basic(ex_w, ex_l, num_filt, kernel_s = 3,stride_l = 1, pool_s =3,
        output_size = 2):
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    x = layers.Conv2D(num_filt, kernel_size=kernel_s, strides=stride_l, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size = pool_s, padding='valid')(x) #valid --> expected (doesn't pad) if strides not specified, then defaults to pool size
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_size, activation='linear')(x)
    
    # don't need to separate if loss calculated on positions at same time (same
    # weights)
    #y1 = layers.Lambda(lambda x: x[0])(denseL)
    #y2 = layers.Lambda(lambda x: x[1])(denseL)
    
    #model = models.Model(inputs, [y1,y2])
    model = models.Model(inputs, outputs)
    model.summary()
    
    return model



#pooling options, size options, etc.
def lunar_net(ex_w, ex_l, num_filt, num_layers, padding_val, kernel_s = 3,stride_l = 1, pool_s =3):
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    for i in range(num_layers):
        
        if i == 0:
            x = layers.Conv2D(num_filt, kernel_size=kernel_s, strides=stride_l,activation='relu', padding=padding_val)(inputs)
        else:
            x = layers.Conv2D(num_filt, kernel_size=kernel_s,strides=stride_l,activation='relu', padding=padding_val)(x)
        
        x = layers.MaxPooling2D(pool_size = pool_s, padding=padding_val)(x) #valid --> expected (doesn't pad) if strides not specified, then defaults to pool size
    
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    model.summary()
    
    return model

def lunar_resnet(ex_w, ex_l, num_filt, num_layers, kernel_s = 3,stride_l = 1,
        pool_s =3, output_size = 2):
    padding_val = 'same'
    # paper version
    #num_res_blocks_per_set = [3,4,6,7]
    #num_filts = [64, 128, 256, 512]
    # first test 2 layers
    num_res_per_layer = [2,2]
    num_filts = [64,128]
    # second test 3 layers
    #num_res_per_layer = [2,2,2]
    #num_filts = [64, 128, 256]
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    
    x = layers.Conv2D(num_filts[0], kernel_size = 7, strides= 1,activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size = 3,strides=2, padding='same')(x)
    for i in range(num_layers):
        for j in range(num_res_per_layer[i]):
            # for first layer of new block after first block, stride is 2 to
            # halve size to account for double the filters
            if i > 0 and j == 0:
                # perform 1x1 conv to project input to higher dimensional space
                # (for more channels), stride of 2 because spatial resolution
                # will be shrunk in 2D convs. 
                saved_x = layers.Conv2D(num_filts[i],kernel_size=1,strides=2,activation='relu',padding='same')(x)
                x = layers.Conv2D(num_filts[i],kernel_size=kernel_s,strides=2,activation='relu',padding='same')(x)
            else:
                saved_x = x
                x = layers.Conv2D(num_filts[i],kernel_size=kernel_s,strides=stride_l,activation='relu',padding='same')(x)
            x = layers.Conv2D(num_filts[i],kernel_size=kernel_s,strides=stride_l,activation='relu',padding='same')(x)
            x = layers.Add()([x, saved_x])

    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Flatten()(x)
    outputs = layers.Dense(output_size, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    model.summary()
    
    return model


def naive_inception_layer(layer_input, nf1,nf3,nf5):
    
    c1 = layers.Conv2D(nf1, kernel_size = 1, strides = 1, activation='relu', padding ='same')(layer_input)
    c3 = layers.Conv2D(nf3, kernel_size = 3, strides = 1,activation='relu',padding='same')(layer_input)
    c5 = layers.Conv2D(nf5, kernel_size = 5, strides = 1, activation = 'relu',padding='same')(layer_input)
    mp3 = layers.MaxPooling2D(pool_size=3,strides=1,padding='same')(layer_input)
    #axis = -1 means that they don't have to be the same shape on that axis (makes sense since diff #s of filters
    layer_output = layers.concatenate([c1,c3,c5,mp3], axis=-1) 

    return layer_output

#less computationally expensive. limits num input channels with 1x1 convs. 
def efficient_inception_layer(layer_in, nf1, nf3_in, nf3_out, nf5_in, nf5_out,nmpf_out):
    c1 =layers.Conv2D(nf1,kernel_size=1,strides=1,activation='relu',padding='same')(layer_in)
    
    c3_i=layers.Conv2D(nf3_in,kernel_size=1,strides=1,activation='relu',padding='same')(layer_in)
    c3 = layers.Conv2D(nf3_out,kernel_size=3,strides=1,activation='relu',padding='same')(c3_i)

    c5_i=layers.Conv2D(nf5_in,kernel_size=1,strides=1,activation='relu',padding='same')(layer_in)
    c5 = layers.Conv2D(nf5_out,kernel_size=5,strides=1,activation='relu',padding='same')(c5_i)

    mp3_i=layers.MaxPooling2D(pool_size=3,strides=1,padding='same')(layer_in)
    mp3 = layers.Conv2D(nmpf_out,kernel_size=1,strides=1,activation='relu',padding='same')(mp3_i)

    layer_output = layers.concatenate([c1,c3,c5,mp3],axis=-1)
    return layer_output

# layer params format: [[nf1_1, nf3_in_1, nf3_out_1, nf5_in_1, nf5_out_1,nmpf_out_1],...,[nf1_n, nf3_in_n, nf3_out_n, nf5_in_n,nf5_out_n,nmpf_out_n]]
# where n is the number of inception layers.

def naive_inception_net(ex_w,ex_l,layer_params):
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    for i in range(len(layer_params)):
        if i == 0:
            x = naive_inception_layer(inputs,*layer_params[i])
        else:
            x = naive_inception_layer(x,*layer_params[i])
    
    x = layers.Flatten()(x)
    output = layers.Dense(2,activation='linear')(x)
    model = models.Model(inputs,output)
    model.summary()
    return model

def efficient_inception_net(ex_w,ex_l,layer_params):
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    for i in range(len(layer_params)):
        if i == 0:
            x = efficient_inception_layer(inputs,*layer_params[i])
        else:
            x = efficient_inception_layer(x,*layer_params[i])
    
    x = layers.Flatten()(x)
    output = layers.Dense(2,activation='linear')(x)
    model = models.Model(inputs,output)
    model.summary()
    return model

def bottleneck_layer(x, expand=128, squeeze=32):
  m = layers.Conv2D(expand, (1,1), padding='same')(x)
  m = layers.Activation('relu')(m)
  m = layers.DepthwiseConv2D((3,3), padding='same')(m)
  m = layers.Activation('relu')(m)
  m = layers.Conv2D(squeeze, (1,1), padding='same')(m)
  
  return layers.Add()([m, x])

def v1_lunar_mobilenet(ex_w, ex_l, num_layers, output_size=2):
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    x = layers.Conv2D(32, kernel_size = 7, strides= 1,activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size = 3,strides=2, padding='same')(x)

    for i in range(num_layers):
        #if i==0:
        #    x = bottleneck_layer(inputs)
        #else:
        #    x = bottleneck_layer(x)
        x = bottleneck_layer(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_size, activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.summary()
    return model

def lunar_mobilenet_base(ex_w, ex_l, output_size=2):
    inputs = layers.Input(shape=(ex_w,ex_l,1))
    base_mobilenet_model = mobilenet_v2.MobileNetV2(input_tensor = inputs,
                                 include_top = False, 
                                 weights = None)
    model = Sequential()
    model.add(base_mobilenet_model)
    #model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(output_size, activation='linear'))
    model.summary()

    return model

def lunar_mobilenet(ex_w, ex_l, num_filt, num_layers, kernel_s = 3,stride_l = 1, pool_s =3, output_size = 2):
    num_res_per_layer = [2,2]
    num_filts = [64,128]
    inputs = layers.Input(shape=(ex_w,ex_l,1))

    x = layers.Conv2D(num_filts[0], kernel_size = 7, strides= 1,activation='relu', padding='same')(inputs)
    
    for i in range(num_layers):
        for j in range(num_res_per_layer[i]):
            if i > 0 and j == 0:
                saved_x = layers.Conv2D(num_filts[i],kernel_size=1,strides=2,activation='relu',padding='same')(x)
                x = layers.Conv2D(num_filts[i],kernel_size=kernel_s,strides=2,activation='relu',padding='same')(x)
            else:
                saved_x = x
                x = layers.Conv2D(num_filts[i],kernel_size=kernel_s,strides=stride_l,activation='relu',padding='same')(x)
            x = layers.Conv2D(num_filts[i],kernel_size=kernel_s,strides=stride_l,activation='relu',padding='same')(x)
            x = layers.Add()([x, saved_x])

    x = layers.Conv2D(num_filts[-1], kernel_size=(1,1), strides = 1, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(output_size, activation='linear')(x)

    model = models.Model(inputs, outputs)
    model.summary()

    return model



