#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from PIL import Image
import numpy as np
import config as cfg
from VGG import VGGNet
import os


# In[2]:


class Transfer(object):
    
    def __init__(self):
        # 获取配置参数
        self.content_img_path = cfg.content_img_path
        self.style_img_path = cfg.style_img_path
        self.vgg_npy_path = cfg.vgg_model_path
        self.learning_rate = float(cfg.learning_rate)
        self.steps = int(cfg.steps)
        self.lambda_c = float(cfg.lambda_c)
        self.lambda_s = float(cfg.lambda_s)
        
        # 获取三种图片
        self.content_value = self.read_image(self.content_img_path)
        self.style_value = self.read_image(self.style_img_path)
        self.result = self.initial_image([1, 224, 224, 3], 127.5, 30)
        
        # 获取占位符
        self.content = tf.placeholder(tf.float32, [1, 224, 224, 3])
        self.style = tf.placeholder(tf.float32, [1, 224, 224, 3])
        
        # 定义三个VGG模型
        data_dict = np.load(self.vgg_npy_path, encoding="latin1", allow_pickle=True).item()
        self.vgg_for_content = VGGNet(data_dict)
        self.vgg_for_style = VGGNet(data_dict)
        self.vgg_for_result = VGGNet(data_dict)
        
        # 构建VGG网络
        self.vgg_for_content.build_vgg(self.content)
        self.vgg_for_style.build_vgg(self.style)
        self.vgg_for_result.build_vgg(self.result)
        
        pass
    
    def initial_image(self, shape, mean, stddev):
        initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)
        pass
    
    def read_image(self, path):
        image = Image.open(path)
#         image = image.resize((224,224))
        np_img = np.array(image)
        np_img = np.asarray([np_img], dtype=np.int32)
        return np_img
    
    def gram_matrix(self, x):
        b, w, h, ch = x.get_shape().as_list()
        feature = tf.reshape(x, [b, w * h, ch])
        gram = tf.matmul(feature, feature, adjoint_a=True) / tf.constant(w * b * ch, tf.float32)
        return gram
        pass
    
    def losses(self):
        # 内容损失
        content_features = [self.vgg_for_content.conv1_2,
                            # self.vgg_for_content.conv2_2
                           ]
        
        result_content_features = [self.vgg_for_result.conv1_2,
                                   # self.vgg_for_result.conv2_2
                                  ]
        
        content_loss = tf.zeros(1, tf.float32)
        for c, c_ in zip(content_features, result_content_features):
            content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])
        
        # 风格损失
        style_features = [self.vgg_for_style.conv4_3]
        
        result_style_features = [self.vgg_for_result.conv4_3]
        
        style_gram = [self.gram_matrix(feature) for feature in style_features]
        
        result_style_gram = [self.gram_matrix(feature) for feature in result_style_features]
        
        style_loss = tf.zeros(1, tf.float32)
        
        for s, s_ in zip(style_gram, result_style_gram):
            style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])
            
        loss = self.lambda_c * content_loss + self.lambda_s * style_loss
        return content_loss, style_loss, loss
        pass
    
    def trans(self):
        content_loss, style_loss, loss = self.losses()
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for step in range(self.steps):
                loss_value, content_loss_value, style_loss_value, _ =                 sess.run([loss, content_loss, style_loss, train_op],
                         feed_dict={
                             self.content: self.content_value,
                             self.style: self.style_value
                         })
                
                print("step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f" 
                      %(step+1, loss_value[0], content_loss_value[0], style_loss_value[0]))
                if (step+1) % 10 == 0:
                    result_img_path = os.path.join("./output_dir", "result-%05d.jpg" %(step+1))
                    result_value = self.result.eval(sess)[0]
                    result_value = np.clip(result_value, 0, 255)
                    img_arr = np.array(result_value, np.uint8)
                    img = Image.fromarray(img_arr)
                    img.save(result_img_path)
        
        pass
    


# In[3]:


transfer = Transfer()
transfer.trans()


# In[ ]:




