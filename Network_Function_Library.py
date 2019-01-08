"""
Network library
@author: Karsten Roth
"""

import cPickle
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import pandas as pd
import os
import time
import theano
from theano import tensor as T
import csv
from scipy import misc as mc
from matplotlib import pyplot as plt
import lasagne
from lasagne.layers import concat,InputLayer, ConcatLayer, Pool2DLayer, Deconv2DLayer, Conv2DLayer, DenseLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, BatchNormLayer, GlobalPoolLayer
from lasagne.nonlinearities import linear, softmax
from lasagne.init import HeUniform
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, BatchNormDNNLayer
from own_layers import TransposedConv3DLayer

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" 3D U-Net """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def upscale_plus_conv_3D(net,no_f_base,f_size,pad,nonlinearity):
    net = lasagne.layers.Upscale3DLayer(net,2)
    net = Conv3DDNNLayer(net,no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain="relu"))
    net = lasagne.layers.PadLayer(net,1)
    return net

#def upscale_NN_3D(net,no_f_base,f_size,pad,nonlinearity):
#    net =
def conv_pool_down_3D(net, no_f_base,f_size,conv_depth,pad,nonlinearity,dropout):
    for i in xrange(conv_depth):
        net = Conv3DDNNLayer(net,no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    if dropout:
        net = DropoutLayer(net,p=dropout)
    return net

def conv_pool_up_3D(net, bs, no_f_base,f_size,conv_depth,pad,nonlinearity,halt=False,useups=False):
    for i in xrange(conv_depth):
        net = Conv3DDNNLayer(net,no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    if not halt:
        #net = Conv3DDNNTransposeLayer(net,no_f_base/2,2,(2,2,2))
        if useups:
            net = upscale_plus_conv_3D(net,no_f_base/2,f_size,pad,nonlinearity)
        else:
            net = TransposedConv3DLayer(net,no_f_base/2,2,(2,2,2))
    return net

def construct_unet_3D(channels=1, no_f_base=8, f_size=3, branches=[2,2,2,2],dropout=0.2,bs=None,
                             class_nums=2, pad="same",nonlinearity=lasagne.nonlinearities.rectify,
                             input_dim=[None,None,None],useups=False):

    net= InputLayer((bs, channels, input_dim[0], input_dim[1], input_dim[2]))

    # Moving downwards the U-shape:
    horizontal_pass=[]
    for i in xrange(len(branches)):
        net = conv_pool_down_3D(net,no_f_base*2**(i),f_size,conv_depth=branches[i],
                             pad=pad,nonlinearity=nonlinearity,dropout=dropout)
        print "Down conv: ",net.output_shape
        horizontal_pass.append(net)
        net = MaxPool3DDNNLayer(net,pool_size=(2,2,2),stride=(2,2,2))
        print "Down Pool: ",net.output_shape

    # Bottleneck
    net = Conv3DDNNLayer(net,no_f_base*2**len(branches),f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    print "Bottleneck conv: ",net.output_shape
    net = Conv3DDNNLayer(net,no_f_base*2**len(branches),f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    print "Bottleneck conv: ",net.output_shape
    #net = Conv3DDNNTransposeLayer(net, no_f_base*2**(len(branches)-1), 2, (2, 2, 2))
    if not useups:
        net = TransposedConv3DLayer(net,no_f_base*2**(len(branches)-1),2,(2,2,2))
    else:
        net = upscale_plus_conv_3D(net,no_f_base*2**(len(branches)-1),f_size,pad,nonlinearity)
    print "Bottleneck up: ",net.output_shape

    # Moving upwards the U-shape:
    for i in xrange(len(branches)):
        print "Pass before concat: ",horizontal_pass[-(i+1)].output_shape
        print "net before concat: ",net.output_shape
        if not useups:
            net = ConcatLayer([net,horizontal_pass[-(i+1)]],cropping=(None,None,"center","center","center"))
        else:
            net = ConcatLayer([net,horizontal_pass[-(i+1)]],cropping=(None,None,"center","center","center"))
        print "Shape after concat: ",net.output_shape
        if i==len(branches)-1:
            net = conv_pool_up_3D(net,bs,no_f_base*2**(len(branches)-1-i),f_size,
                           pad=pad,nonlinearity=nonlinearity,conv_depth=branches[i],halt=True,useups=False)
        else:
            net = conv_pool_up_3D(net,bs,no_f_base*2**(len(branches)-1-i),f_size,
                           pad=pad,nonlinearity=nonlinearity,conv_depth=branches[i],halt=False,useups=False)
        print "Conv up: ",net.output_shape
    # Class layer: Work around standard softmax bc. it doesn't work with tensor4/3.
    # Hence, we reshape and feed it to an external Nonlinearity layer.
    # net["class_ns"] is the output in image-related shape.
    imageout = net  = Conv3DDNNLayer(net, class_nums, 1, nonlinearity=linear,W=lasagne.init.HeNormal(gain='relu'))
    print "imageout shape: ",net.output_shape
    net  = DimshuffleLayer(net, (1, 0, 2, 3, 4))
    print "After shuffle shape: ",net.output_shape
    net  = ReshapeLayer(net, (class_nums, -1))
    print "Reshape shape: ",net.output_shape
    net  = DimshuffleLayer(net, (1, 0))
    print "Dimshuffle shape: ",net.output_shape
    # Flattened output to be able to feed it to lasagne.objectives.categorical_crossentropy.
    net  = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)
    #imout = NonlinearityLayer(imageout,nonlinearity=lasagne.nonlinearities.softmax)
    return net,imageout
    del net, imageout,imout

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Dense-Net """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def sub_block(net,  no_f_base, f_size_base, dropout, give_name, do_relu=True):
    net = BatchNormLayer(net, name=give_name+"_bnorm")
    if do_relu:
        net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.rectify, name=give_name+"_relu")
    net = Conv2DLayer(net, no_f_base, f_size_base, pad="same",W=lasagne.init.HeNormal(gain="relu"),b=None, name=give_name+"_conv")
    if dropout:
        net = DropoutLayer(net, dropout)
    return net

def block_transit(net, drop_p, name_to_give, pooltype="average_inc_pad"):
    net = sub_block(net, no_f_base=net.output_shape[1], f_size_base=1, dropout=drop_p, give_name=name_to_give, do_relu=False)
    net = Pool2DLayer(net, 2, mode=pooltype, name=name_to_give+"_avgpool")
    return net

def block_create(net, n_layer, k, drop_p, name_to_give):
    for i in xrange(n_layer):
        net_conv = sub_block(net, no_f_base=k, f_size_base=3, dropout=drop_p, give_name=name_to_give+"_l%02d" % (i + 1))
        net = ConcatLayer([net, net_conv], axis=1, name=name_to_give+ "_l%02d_join" % (i + 1))
    return net

def construct_densenet(channels=1, no_f_base=16, f_size_base=3, bs=None, class_nums=2, k=12, denseblocks=[4,4,4,4], dropout=0, input_var=None, pad="same",
                       c_nonlinearity=lasagne.nonlinearities.rectify, f_nonlinearity=lasagne.nonlinearities.softmax, input_dim=[112,112]):
    #Network start
    net= InputLayer((bs, channels, input_dim[0], input_dim[1]), input_var, name="input")
    net= Conv2DLayer(net, no_f_base, f_size_base, pad=pad, W=lasagne.init.HeNormal(gain='relu'), b=None, nonlinearity=None, name="pre_conv")
    if dropout:
        net = DropoutLayer(net, dropout)
    #Block building
    for blocks in xrange(len(denseblocks)):
        net = block_create(net, denseblocks[blocks], k, drop_p=dropout, name_to_give="block%d" % (blocks + 1))
        if blocks<len(denseblocks)-1:
            net = block_transit(net, drop_p=dropout, pooltype="average_inc_pad", name_to_give="block%d_transit" % (blocks + 1))
    #Finishing off
    net = BatchNormLayer(net, name="final_bnorm")
    net = NonlinearityLayer(net, nonlinearity=c_nonlinearity,name="final_relu")
    net = GlobalPoolLayer(net, name="global_pool")
    net = DenseLayer(net, class_nums, nonlinearity=f_nonlinearity, name="out")

    return net


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Tiramisu (U_net + Dense_net) Author-version"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):

    l = NonlinearityLayer(BatchNormLayer(inputs))
    l = Conv2DLayer(l, n_filters, filter_size, pad='same', W=HeUniform(gain='relu'), nonlinearity=linear,
                    flip_filters=False)
    if dropout_p != 0.0:
        l = DropoutLayer(l, dropout_p)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = Pool2DLayer(l, 2, mode='max')
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    l = ConcatLayer(block_to_upsample)
    l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])
    return l

def SoftmaxLayer(inputs, n_classes):
    l = Conv2DLayer(inputs, n_classes, filter_size=1, nonlinearity=linear, W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)
    l = DimshuffleLayer(l,(1,0,2,3))
    l = ReshapeLayer(l, (n_classes,-1))
    l = DimshuffleLayer(l, (1,0))
    l = NonlinearityLayer(l, nonlinearity=softmax)
    return l

def construct_tiramisu_author(channels=1, no_f_base=45, f_size_base=3, bs=None, class_nums=2, k=16, denseblocks=[4,5,7,10,12], blockbottom=15, dropout_p=0.2, input_var=None, pad="same",c_nonlinearity=lasagne.nonlinearities.rectify,
                   f_nonlinearity=lasagne.nonlinearities.rectify, input_dim=[112,112]):

    #Network start
    inputs = InputLayer((bs, channels, input_dim[0], input_dim[1]), input_var)
    stack = Conv2DLayer(inputs, no_f_base, f_size_base, pad=pad, W=HeUniform(gain='relu'), flip_filters=False)
    n_filters = no_f_base

    #Downward block building
    horizontal_pass=[]
    for blocks in xrange(len(denseblocks)):
        for j in xrange(denseblocks[blocks]):
            l = BN_ReLU_Conv(stack, k, dropout_p=dropout_p)
            stack = ConcatLayer([stack, l])
            print stack.output_shape
            n_filters += k
        horizontal_pass.append(stack)
        stack = TransitionDown(stack, n_filters, dropout_p)
        print "After TransitionDown: ",stack.output_shape
    print "---Down done---"

    #Bottom dense block
    block_to_upsample = []

    for bottom in xrange(blockbottom):
        l = BN_ReLU_Conv(stack, k, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = ConcatLayer([stack, l])
        print "Bottomblock: ",stack.output_shape
    print "---Bottom done---"

    #Up dense block
    for block in xrange(len(denseblocks)):
        n_filters_keep = k*denseblocks[block]
        stack = TransitionUp(horizontal_pass[-(block+1)], block_to_upsample, n_filters_keep)
        print "After TransitionUp: ",stack.output_shape
        block_to_upsample = []
        for j in xrange(denseblocks[-(block+1)]):
            l = BN_ReLU_Conv(stack, k, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = ConcatLayer([stack, l])
        print stack.output_shape
    print "---Up done---"
    #Out block
    image_out = Conv2DLayer(stack, class_nums, 1, pad=pad, W=lasagne.init.HeUniform(gain='relu'), nonlinearity=linear)
    net = SoftmaxLayer(stack, class_nums)

    return net,image_out

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Tiramisu (U_net + Dense_net) """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def tiramisu_layer(net,  no_f_base, f_size_base, dropout):
    net = BatchNormLayer(net)
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.rectify)
    net = Conv2DLayer(net, no_f_base, f_size_base, pad="same",W=lasagne.init.HeUniform(gain="relu"),b=None, flip_filters=False)
    if dropout:
        net = DropoutLayer(net, dropout)
    return net

def tiramisu_denseblock(net, n_layer, k, drop_p):
    conv_layer_coll=[]
    for i in xrange(n_layer):
        net_conv = tiramisu_layer(net, no_f_base=k, f_size_base=3, dropout=drop_p)
        if i<n_layer-1:
            net = ConcatLayer([net, net_conv], axis=1)
        conv_layer_coll.append(net_conv)
    net = ConcatLayer(conv_layer_coll, axis=1)
    return net
#Note: we do not simply repeat concatenation s.t. we can apply the same dense block method whil transitioning upwards. We concat the input after the dense block.

def tiramisu_transistion_down(net, drop_p, pooltype="average_inc_pad"):
    net = BatchNormLayer(net)
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.rectify)
    net = Conv2DLayer(net, net.output_shape[1], 1, pad="same",W=lasagne.init.HeUniform(gain="relu"),b=None,flip_filters=False)
    if drop_p:
        net = DropoutLayer(net, drop_p)
    net = Pool2DLayer(net, 2)
    return net

def tiramisu_transistion_up(net, f_size_base, stride):
    net = Deconv2DLayer(net, net.output_shape[1], f_size_base, stride)
    return net

def tiramisu_transition_bottom(net, drop_p, pooltype="average_inc_pad"):
    net = tiramisu_layer(net, no_f_base=net.output_shape[1], f_size_base=1, dropout=drop_p)
    return net

def construct_tiramisu(channels=1, no_f_base=45, f_size_base=3, bs=None, class_nums=2, k=16, denseblocks=[4,5,7,10,12], blockbottom=[15],
                       dropout=0, input_var=None, pad="same",c_nonlinearity=lasagne.nonlinearities.rectify,
                       f_nonlinearity=lasagne.nonlinearities.rectify, input_dim=[112,112]):

    #Network start
    net = InputLayer((bs, channels, input_dim[0], input_dim[1]), input_var)
    net = Conv2DLayer(net, no_f_base, f_size_base, pad=pad, W=lasagne.init.HeUniform(gain='relu'), flip_filters=False)
    #Downward block building
    horizontal_pass=[]
    for blocks in xrange(len(denseblocks)):
        net_preblock = net
        net = tiramisu_denseblock(net, denseblocks[blocks], k, drop_p=dropout)
        net = ConcatLayer([net_preblock, net], axis=1) #Connection around Denseblock
        print "Input shape: {}, after concat: {}".format(net_preblock.output_shape,net.output_shape)
        horizontal_pass.append(net)
        net = tiramisu_transistion_down(net, drop_p=dropout)
        print "After transition down: ",net.output_shape
    print "---Down done---"
    #Bottom dense block
    for bottom in xrange(len(blockbottom)):
        net = tiramisu_denseblock(net, blockbottom[bottom], k, drop_p=dropout)
        if bottom < len(blockbottom)-1:
            net = tiramisu_transition_bottom(net, drop_p=dropout)
        print "Bottom: ",net.output_shape
    print "---Bottom done---"
    #Up dense block
    for block in xrange(len(denseblocks)):
        print "Before concat size: ",net.output_shape
        net = tiramisu_transistion_up(net, f_size_base, 2)
        net = ConcatLayer([net, horizontal_pass[-(block+1)]], axis=1,cropping=[None, None, "center", "center"])
        print "After concat size: ",net.output_shape
        net = tiramisu_denseblock(net, denseblocks[-(block+1)], k, drop_p=dropout)
        print "Denseblock size: ",net.output_shape
    print "---Up done---"
    #Out block
    image_out = net = Conv2DLayer(net, class_nums, 1, pad=pad, W=lasagne.init.HeUniform(gain='relu'), nonlinearity=linear, flip_filters=False)
    net = DimshuffleLayer(net,(1,0,2,3))
    net = ReshapeLayer(net, (class_nums,-1))
    net = DimshuffleLayer(net, (1,0))
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)

    return net,image_out
    net = None
    image_out=None

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" U-net general structure """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def construct_unet(channels=1, no_f_base=8, f_size=3, dropout=False, bs=None, class_nums=2, pad="same",nonlinearity=lasagne.nonlinearities.rectify, input_dim=[512,512]):
    net={}
    net["input"]= InputLayer(shape=(bs, channels, input_dim[0], input_dim[1]))

    # Moving downwards the U-shape. Simplified:
    net["conv_down11"] = Conv2DLayer(net["input"],no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_down12"] = Conv2DLayer(net["conv_down11"],no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool1"]      = Pool2DLayer(net["conv_down12"],pool_size=2)

    net["conv_down21"] = Conv2DLayer(net["pool1"],no_f_base*2,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_down22"] = Conv2DLayer(net["conv_down21"],no_f_base*2,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool2"]      = Pool2DLayer(net["conv_down22"],pool_size=2)

    net["conv_down31"] = Conv2DLayer(net["pool2"],no_f_base*4,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_down32"] = Conv2DLayer(net["conv_down31"],no_f_base*4,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool3"]      = Pool2DLayer(net["conv_down32"],pool_size=2)

    net["conv_down41"] = Conv2DLayer(net["pool3"],no_f_base*8,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_down42"] = Conv2DLayer(net["conv_down41"],no_f_base*8,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    to_drop1 = net["pool4"]      = Pool2DLayer(net["conv_down42"],pool_size=2)

    if dropout:
        to_drop1 = DropoutLayer(to_drop1, p=0.5)

    #vvvv bottom vvvv
    net["conv_bottom1"] = Conv2DLayer(to_drop1,no_f_base*16,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_bottom2"] = Conv2DLayer(net["conv_bottom1"],no_f_base*16,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["deconv_bottom1"]      = Deconv2DLayer(net["conv_bottom2"], no_f_base*8, 2, 2)
    #^^^^ bottom ^^^^

    # Moving upwards the U-shape. Simplified:
    net["concat1"] = concat([net["deconv_bottom1"], net["conv_down42"]], cropping=(None, None, "center", "center"))
    net["conv_up11"]= Conv2DLayer(net["concat1"], no_f_base*8, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_up11"]= Conv2DLayer(net["conv_up11"], no_f_base*8, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["deconv_up1"] = Deconv2DLayer(net["conv_up11"], no_f_base*4, 2, 2)

    net["concat2"] = concat([net["deconv_up1"], net["conv_down32"]], cropping=(None, None, "center", "center"))
    net["conv_up21"]= Conv2DLayer(net["concat2"], no_f_base*4, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_up22"]= Conv2DLayer(net["conv_up21"], no_f_base*4, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["deconv_up2"] = Deconv2DLayer(net["conv_up22"], no_f_base*2, 2, 2)

    net["concat3"] = concat([net["deconv_up2"], net["conv_down22"]], cropping=(None, None, "center", "center"))
    net["conv_up31"]= Conv2DLayer(net["concat3"], no_f_base*2, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_up32"]= Conv2DLayer(net["conv_up31"], no_f_base*2, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["deconv_up3"] = Deconv2DLayer(net["conv_up32"], no_f_base, 2, 2)

    net["concat4"] = concat([net["deconv_up3"], net["conv_down12"]], cropping=(None, None, "center", "center"))
    net["conv_up41"]= Conv2DLayer(net["concat4"], no_f_base, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_up42"]= Conv2DLayer(net["conv_up41"], no_f_base, f_size, pad=pad, nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    # Class layer: Work around standard softmax bc. it doesn't work with tensor4/3.
    # Hence, we reshape and feed it to an external Nonlinearity layer.
    # net["class_ns"] is the output in image-related shape.

    net["out"] = Conv2DLayer(net["conv_up42"], class_nums, 1, nonlinearity=None,W=lasagne.init.HeNormal(gain='relu'))
    net["layer_shuffle_dim"] = DimshuffleLayer(net["out"], (1, 0, 2, 3))
    net["reshape_layer"] = ReshapeLayer(net["layer_shuffle_dim"], (class_nums, -1))
    net["layer_shuffle_dim2"] = DimshuffleLayer(net["reshape_layer"], (1, 0))
    # Flattened output to be able to feed it to lasagne.objectives.categorical_crossentropy.
    net["out_optim"] = NonlinearityLayer(net["layer_shuffle_dim2"], nonlinearity=lasagne.nonlinearities.softmax)

    return net
    net = None

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" U-net recursive """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def conv_pool_down(net, no_f_base,f_size,conv_depth,pad,nonlinearity,dropout):
    for i in xrange(conv_depth):
        net = Conv2DLayer(net,no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    if dropout:
        net = DropoutLayer(net,p=dropout)
    return net

def conv_pool_up(net, no_f_base,f_size,conv_depth,pad,nonlinearity,halt=False):
    for i in xrange(conv_depth):
        net = Conv2DLayer(net,no_f_base,f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    if not halt:
        net = Deconv2DLayer(net,no_f_base/2,2,2)
    return net

def construct_unet_recursive(channels=1, no_f_base=8, f_size=3, branches=[2,2,2,2],dropout=0.2,bs=None,
                             class_nums=2, pad="same",nonlinearity=lasagne.nonlinearities.rectify, input_dim=[512,512]):

    net= InputLayer((bs, channels, input_dim[0], input_dim[1]))
    # Moving downwards the U-shape:
    horizontal_pass=[]
    for i in xrange(len(branches)):
        net = conv_pool_down(net,no_f_base*2**(i),f_size,conv_depth=branches[i],
                             pad=pad,nonlinearity=nonlinearity,dropout=dropout)
        horizontal_pass.append(net)
        net = Pool2DLayer(net,pool_size=2)
    # Bottleneck
    net = Conv2DLayer(net,no_f_base*2**len(branches),f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net = Conv2DLayer(net,no_f_base*2**len(branches),f_size,pad=pad,nonlinearity=nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net = Deconv2DLayer(net, no_f_base*2**(len(branches)-1), 2, 2)
    # Moving upwards the U-shape:
    for i in xrange(len(branches)):
        net = PadLayer(net,1)
        net = ConcatLayer([net,horizontal_pass[-(i+1)]],cropping=(None,None,"center","center"))
        if i==len(branches)-1:
            net = conv_pool_up(net,no_f_base*2**(len(branches)-1-i),f_size,
                           pad=pad,nonlinearity=nonlinearity,conv_depth=branches[i],halt=True)
        else:
            net = conv_pool_up(net,no_f_base*2**(len(branches)-1-i),f_size,
                           pad=pad,nonlinearity=nonlinearity,conv_depth=branches[i],halt=False)

    # Class layer: Work around standard softmax bc. it doesn't work with tensor4/3.
    # Hence, we reshape and feed it to an external Nonlinearity layer.
    # net["class_ns"] is the output in image-related shape.
    imageout = net  = Conv2DLayer(net, class_nums, 1, nonlinearity=linear,W=lasagne.init.HeNormal(gain='relu'))
    net  = DimshuffleLayer(net, (1, 0, 2, 3))
    net  = ReshapeLayer(net, (class_nums, -1))
    net  = DimshuffleLayer(net, (1, 0))
    # Flattened output to be able to feed it to lasagne.objectives.categorical_crossentropy.
    net  = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)

    return net,imageout
    del net, imageout

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" VGG16-net """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def construct_vgg16net(channels=1, no_f_base=64, f_size=3, dropout=False, bs=None, num_classes=3, pad=1,c_nonlinearity=lasagne.nonlinearities.rectify,f_nonlinearity=lasagne.nonlinearities.tanh, input_dim=[512,512], flip=False, fcn=4096):
    net = {}
    net["input"] = InputLayer(shape=(bs, channels, input_dim[0], input_dim[1]))
    net["conv_11"] = Conv2DLayer(net["input"], no_f_base, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_12"] = Conv2DLayer(net["conv_11"], no_f_base, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool1"] = Pool2DLayer(net["conv_12"], 2)
    net["conv_21"] = Conv2DLayer(net["pool1"], no_f_base*2, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_22"] = Conv2DLayer(net["conv_21"], no_f_base*2, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool2"] = Pool2DLayer(net["conv_22"], 2)
    net["conv_31"] = Conv2DLayer(net["pool2"], no_f_base*4, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_32"] = Conv2DLayer(net["conv_31"], no_f_base*4, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_33"] = Conv2DLayer(net["conv_32"], no_f_base*4, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool3"] = Pool2DLayer(net["conv_33"], 2)
    net["conv_41"] = Conv2DLayer(net["pool3"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_42"] = Conv2DLayer(net["conv_41"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_43"] = Conv2DLayer(net["conv_42"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool4"] = Pool2DLayer(net["conv_43"], 2)
    net["conv_51"] = Conv2DLayer(net["pool4"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_52"] = Conv2DLayer(net["conv_51"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["conv_53"] = Conv2DLayer(net["conv_52"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity,W=lasagne.init.HeNormal(gain='relu'))
    net["pool5"] = Pool2DLayer(net["conv_53"], 2)

    net["full_con1"] = DenseLayer(net["pool5"], num_units=fcn, nonlinearity=f_nonlinearity)
    net["drop_full_con1"] = DropoutLayer(net["full_con1"], p=0.5)
    net["full_con2"] = DenseLayer(net["drop_full_con1"], num_units=fcn, nonlinearity=f_nonlinearity)
    net["drop_full_con2"] = DropoutLayer(net["full_con2"], p=0.5)
    net["full_con3"] = DenseLayer(net["drop_full_con2"], num_units=num_classes, nonlinearity=None)
    net["out"] = NonlinearityLayer(net["full_con3"], nonlinearity=lasagne.nonlinearities.softmax)
    return net
    del net


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" VGG13-net """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def construct_vgg13net(channels=1, no_f_base=64, f_size=3, dropout=False, bs=None, num_classes=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify, input_dim=[512,512], flip=True, fcn=4096):
    net = {}
    net["input"] = InputLayer(shape=(bs, channels, input_dim[0], input_dim[1]))
    net["conv_11"] = Conv2DLayer(net["input"], no_f_base, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["conv_12"] = Conv2DLayer(net["conv_11"], no_f_base, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["pool1"] = Pool2DLayer(net["conv_12"], 2)
    net["conv_21"] = Conv2DLayer(net["pool1"], no_f_base*2, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["conv_22"] = Conv2DLayer(net["conv_21"], no_f_base*2, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["pool2"] = Pool2DLayer(net["conv_22"], 2)
    net["conv_31"] = Conv2DLayer(net["pool2"], no_f_base*4, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["conv_32"] = Conv2DLayer(net["conv_31"], no_f_base*4, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["conv_33"] = Conv2DLayer(net["conv_32"], no_f_base*4, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["pool3"] = Pool2DLayer(net["conv_33"], 2)
    net["conv_41"] = Conv2DLayer(net["pool3"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["conv_42"] = Conv2DLayer(net["conv_41"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["conv_43"] = Conv2DLayer(net["conv_42"], no_f_base*8, f_size, pad=pad, flip_filters=flip, nonlinearity=nonlinearity)
    net["pool4"] = Pool2DLayer(net["conv_43"], 2)

    drop1 = net["full_con1"] = DenseLayer(net["pool4"], num_units=fcn, nonlinearity=nonlinearity)
    if dropout:
        drop1 = DropoutLayer(drop1, p=0.5)
    drop2 = net["full_con2"] = DenseLayer(drop1, num_units=fcn, nonlinearity=nonlinearity)
    if dropout:
        drop2 = DropoutLayer(drop2, p=0.5)
    net["full_con3"] = DenseLayer(drop2, num_units=num_classes, nonlinearity=None)
    net["out"] = NonlinearityLayer(net["full_con3"], nonlinearity=lasagne.nonlinearities.softmax)
    return net
    del net


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Small net for small data sets """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Note: Occupies a buttload of memory for bigger data sets.
def construct_simplenet(channels=1, no_f_base=8, f_size=3, dropout=True, bs=None, num_classes=3, pad=1,c_nonlinearity=lasagne.nonlinearities.rectify,f_nonlinearity=lasagne.nonlinearities.tanh, input_dim=[1024,768], flip=True, fcn=300):
    net = {}
    net["input"] = InputLayer(shape=(bs, channels, input_dim[0], input_dim[1]))
    net["conv_11"] = Conv2DLayer(net["input"], no_f_base, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity)
    net["conv_12"] = Conv2DLayer(net["conv_11"], no_f_base, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity)
    net["pool1"] = Pool2DLayer(net["conv_12"], 2)
    net["conv_21"] = Conv2DLayer(net["pool1"], no_f_base*2, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity)
    net["conv_22"] = Conv2DLayer(net["conv_21"], no_f_base*2, f_size, pad=pad, flip_filters=flip, nonlinearity=c_nonlinearity)
    net["pool2"] = Pool2DLayer(net["conv_22"], 2)
    net["pool2"].output_shape
    drop1 = net["full_con1"] = DenseLayer(net["pool2"], num_units=fcn, nonlinearity=f_nonlinearity)
    if dropout:
        drop1 = DropoutLayer(drop1, p=0.5)
    drop2 = net["full_con2"] = DenseLayer(drop1, num_units=fcn, nonlinearity=f_nonlinearity)
    if dropout:
        drop2 = DropoutLayer(drop2, p=0.5)
    net["full_con3"] = DenseLayer(drop2, num_units=num_classes, nonlinearity=None)
    net["out"] = NonlinearityLayer(net["full_con3"], nonlinearity=lasagne.nonlinearities.softmax)
    return net
    del net

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Normal ConvNet recursive """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Note: Occupies a buttload of memory for bigger data sets.
def conv_block(net,no_f_base,f_size,pad,flip,nonlinearity,depth):
    for i in xrange(depth):
        net = Conv2DLayer(net,no_f_base,f_size,pad=pad,flip_filters=flip,nonlinearity=nonlinearity)
    net = Pool2DLayer(net,pool_size=2)
    return net
def fcn_block(net,num_units,nonlinearity,dropout):
    for i in num_units:
        net = DenseLayer(net, num_units=i,nonlinearity=nonlinearity)
        if dropout:
            net = DropoutLayer(net,p=dropout)
    return net
def construct_simplenet_recursive(channels=1, no_f_base=8, f_size=3, bs=None, conv_list=[2,2,2],fcn_list=[300,300],num_classes=3,
                        pad=1,c_nonlinearity=lasagne.nonlinearities.rectify,flip=False,dropout=0,
                        f_nonlinearity=lasagne.nonlinearities.tanh, input_dim=[512,512]):

    net = InputLayer(shape=(bs, channels, input_dim[0], input_dim[1]))
    for i in conv_list:
        net = conv_block(net,no_f_base,f_size,pad,flip,nonlinearity,i)
    net = fcn_block(net,fcn_list,nonlinearity,dropout)
    net = DenseLayer(drop2, num_units=num_classes, nonlinearity=None)
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)
    return net
    del net

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" ResNet """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def construct_resnet(channels=1, no_f_base=8, f_size=3, dropout=0.2, bs=None, num_classes=3):
    print "hi"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Image processing functions """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Data whitening
def whiten(X, eps=1e-10):
    diag,V = np.linalg.eigh(np.dot(X.T,X))
    D      = np.diag(1./np.sqrt(diag+eps))
    Whitmat= np.dot(np.dot(V,D),V.T)
    X_white= np.dot(X,Whitmat)
    return X_white
