# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:52:07 2019

@author: Nomula Dheeraj Kumar
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)

with tf.name_scope("inputs"):
    x=tf.placeholder(tf.float32,[None,28*28],name='X')
    x_reshaped=tf.reshape(x,shape=[-1,28,28,1])
    y=tf.placeholder(tf.float32,[None,10],name='Y')
with tf.name_scope("Layer1"):
#    padding=SAME means we shall 0 pad the input such a way that output x,y dimensions are same as that of input.
#   filters= [filter_size , filter_size , num_input_channels , num_filters]
#    strides= defines how much you move your filter when doing convolution. 
#    In this function, it needs to be a Tensor of size>=4 
#    i.e. [batch_stride x_stride y_stride depth_stride]. 
#    batch_stride is always 1 as you don’t want to skip images in your batch.
#    x_stride and y_stride are same mostly and the choice is part of network design and we shall use them as 1 i
#    n our example. depth_stride is always set as 1 as you don’t skip along the depth.
    conv1=tf.layers.conv2d(inputs=x_reshaped,filters=24,kernel_size=[5,5],padding='SAME',name="conv1")
    conv1=tf.nn.relu(conv1)
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool1")
with tf.name_scope("Layer2"):
    conv2=tf.layers.conv2d(inputs=conv1,filters=36,kernel_size=[5,5],padding="same",name='conv2')
    conv2=tf.nn.relu(conv2)
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool2")
with tf.name_scope("Layer3"):
    conv3=tf.layers.conv2d(inputs=conv2,filters=48,kernel_size=[5,5],padding='same',name='conv3')
    conv3=tf.nn.relu(conv3)
    conv3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool3")
with tf.name_scope("Layer4"):
    conv4=tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[5,5],padding='same',name='conv4')
    conv4=tf.nn.relu(conv4)
    conv4=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name="pool4")
with tf.name_scope("Fc1"):
    fc1=tf.contrib.layers.flatten(conv4,)
    fc1=tf.layers.dense(fc1,24,name='fc1')
#    fc1=tf.layers.batch_normalization(fc1)
    fc1=tf.nn.relu(fc1)
with tf.name_scope("Output"):
    fc2=tf.layers.dense(fc1,10,name='Output')
    output=tf.nn.softmax(fc2,name='Y_out')
    
with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
    
init=tf.global_variables_initializer()
n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        epoch_loss=0
        for iteration in range(mnist.train.num_examples//batch_size):
            epoch_x,epoch_y=mnist.train.next_batch(batch_size)
            _,c=sess.run([training_op,loss],feed_dict={x:epoch_x,y:epoch_y})
            epoch_loss+=c
        print('Epoch',epoch,'completed out of',n_epochs,'loss:',epoch_loss)
    correct=tf.equal(tf.argmax(output,axis=1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct,'float'))
    print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

        
        