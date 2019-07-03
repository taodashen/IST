#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import config as  cfg


# In[4]:


VGG_MEAN = [103.939, 116.779, 123.68]


class VGGNet(object):
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0])
    
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0])
    
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1])
    
    def conv_layer(self, x, name):
        with tf.name_scope(name):
            w = self.get_conv_filter(name)
            b = self.get_bias(name)
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.nn.bias_add(conv, b)
            conv = tf.nn.relu(conv)
            return conv
            
    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name=name)
            
        
    def build_vgg(self, inputs):
        """创建vgg16的模型
        Param:
        - inputs [1, 224, 224, 3]
        """
        
        print("构建模型中")
        start = time.time()
        
        r, g, b = tf.split(inputs, [1, 1, 1], axis=3)
        x_bgr = tf.concat([
            b - VGG_MEAN[0],
            g - VGG_MEAN[1],
            r - VGG_MEAN[2]], axis=3)
        
        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        self.conv1_1 = self.conv_layer(x_bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")
        
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")
        
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, "pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, "pool4")
        
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, "pool5")
        
        print("构建完成，共花费:%4ds" %(time.time() - start))
        

