# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.network.base_net import BaseNet



class ClassifyNetMultiOutput(BaseNet):
    """
    a simple classification network
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='classify_net_multioutput'):

        super(ClassifyNetMultiOutput, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3, 'stride': 1},
            {'name': 'conv_1', 'n_features': 16, 'kernel_size': 3, 'stride': 2},
            {'name': 'conv_2', 'n_features': 32, 'kernel_size': 3, 'stride': 1},
            {'name': 'conv_3', 'n_features': 32, 'kernel_size': 3, 'stride': 2},
            {'name': 'conv_4', 'n_features': 64, 'kernel_size': 3, 'stride': 1},
            {'name': 'conv_5', 'n_features': 64, 'kernel_size': 3, 'stride': 2},
            {'name': 'conv_1d','n_features':  2, 'kernel_size': 1, 'stride': 1}]

    def layer_op(self, images, is_training, layer_id=-1):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ### Resize
        images_down = tf.image.resize_bilinear(images, (18,18,6))

        ### first convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        # Check output
        flow3 = first_conv_layer(images, is_training)

        ### second convolution layer
        params = self.layers[1]
        second_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = second_conv_layer(flow, is_training)
        layer_instances.append((second_conv_layer, flow))

        ### third convolution layer
        params = self.layers[2]
        third_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = third_conv_layer(flow, is_training)
        layer_instances.append((third_conv_layer, flow))

        ### fourth convolution layer
        params = self.layers[3]
        fourth_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fourth_conv_layer(flow, is_training)
        layer_instances.append((fourth_conv_layer, flow))

        ### fifth convolution layer
        params = self.layers[4]
        fifth_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fifth_conv_layer(flow, is_training)
        layer_instances.append((fifth_conv_layer, flow))

        ### sixth convolution layer
        params = self.layers[5]
        sixth_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = sixth_conv_layer(flow, is_training)
        layer_instances.append((sixth_conv_layer, flow))

        ### 1d convolution layer
        params = self.layers[6]
        conv_layer_1d = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            acti_func=None,
            with_bias=False,
            with_bn=False,
            w_initializer=None,
            w_regularizer=None,
            b_initializer=None,
            b_regularizer=None,
            name=params['name'])

        print('\nAfter convolutions:\n', flow)

        ### Flow2 1d conv
        flow2 = conv_layer_1d(flow, is_training)
        layer_instances.append((conv_layer_1d, flow2))
        flow2 = tf.sigmoid(flow2)
        print('\nFlow2 after 1d conv:\n', flow2)

        ### max reduction
        print('\nBefore mean reduction\n', flow)
        flow = tf.reduce_max(flow, axis=[1, 2, 3], keepdims=True)
        #flow = tf.reduce_mean(flow, axis=[1, 2, 3], keepdims=True)
        print('\nAfter mean reduction\n', flow)

        ### Flow1 1d conv
        flow1 = conv_layer_1d(flow, is_training)
        layer_instances.append((conv_layer_1d, flow1))
        flow1 = tf.layers.flatten(flow1)
        print('\nFlow1 after 1d conv:\n', flow1)

        # set training properties
        # if is_training:
        #     self._print(layer_instances)
        #     return layer_instances[-1][1]
        print('\nEntered ClassifyNetMultiOutput\n')
        print('\nFlow1:\n', flow1)
        print('\nFlow2:\n', flow2)
        return flow1, flow2, flow3

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
