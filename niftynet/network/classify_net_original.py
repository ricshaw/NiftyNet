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



class ClassifyNet(BaseNet):
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
                 name='classify_net'):

        super(ClassifyNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 8, 'kernel_size': 3, 'stride': 1},
            {'name': 'conv_1', 'n_features': 8, 'kernel_size': 3, 'stride': 2},
            {'name': 'conv_2', 'n_features': 16, 'kernel_size': 3, 'stride': 1},
            {'name': 'conv_3', 'n_features': 16, 'kernel_size': 3, 'stride': 2},
            {'name': 'conv_4', 'n_features': 24, 'kernel_size': 3, 'stride': 1},
            {'name': 'conv_5', 'n_features': 24, 'kernel_size': 3, 'stride': 2},
            {'name': 'mp', 'kernel_size': 1, 'stride': 1}]
        #num_classes

    def layer_op(self, images, is_training, layer_id=-1):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

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

        ### 1x1x1 convolution layer
        #params = self.layers[6]
        #fc_layer = ConvolutionalLayer(
        #    n_output_chns=params['n_features'],
        #    kernel_size=params['kernel_size'],
        #    stride=params['stride'],
        #    acti_func=self.acti_func,
        #    w_initializer=self.initializers['w'],
        #    w_regularizer=self.regularizers['w'],
        #    name=params['name'])
        #flow = fc_layer(flow, is_training)
        #layer_instances.append((fc_layer, flow))

        ### max pooling layer
        params = self.layers[6]
        mp_layer = DownSampleLayer(
            func='MAX',
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            name=params['name'])
        flow = mp_layer(flow)
        layer_instances.append((mp_layer, flow))

        # set training properties
        if is_training:
            self._print(layer_instances)
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
