#!/usr/bin/env python
# coding: utf-8

#This file reads .npy file (X, Ycluster) then uses 3 layers neural net to try to train a classifier.

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


Xtrain = np.load("Xtrain.npy")
Yclustertrain = np.load("Yclustertrain.npy")


# In[3]:


print(Xtrain.shape)
print(Yclustertrain.shape)


# In[4]:


X = tf.placeholder(dtype = tf.float32, shape = Xtrain.shape, name="Xtrain")
Y = tf.placeholder(dtype = tf.float32, shape = Yclustertrain.shape, name = "Yclustertrain")


# In[11]:


#h = softmax(relu(X*m1 + b1)*m2 + b2)
inshape = 11
intermediate = 11*6
outshape = 19
m1 = tf.Variable(tf.random.normal((inshape, intermediate), dtype = tf.float32, stddev=0.01), name = "m1")
b1 = tf.Variable(tf.random.normal((1, intermediate), dtype = tf.float32, stddev=0.01), name = "b1")
m2 = tf.Variable(tf.random.normal((intermediate, outshape), dtype = tf.float32, stddev=0.01), name = "m2")
b2 = tf.Variable(tf.random.normal((1, outshape), dtype = tf.float32, stddev=0.01), name = "b2")

h = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(X, m1) + b1), m2) + b2, axis = 1, name = "softmax")
print(h)


# In[22]:


loss = tf.reduce_mean(-(Y*tf.log(h) + (1-Y)*tf.log(1-h)), name = "loss")


# In[34]:


optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
trainer = optimizer.minimize(loss, name = "trainer")


# In[47]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(600):
        _, lossvalue = sess.run([trainer, loss], feed_dict = {X:Xtrain, Y:Yclustertrain})
        print(lossvalue)
    np.save("m1.npy", sess.run(m1))
    np.save("b1.npy", sess.run(b1))
    np.save("m2.npy", sess.run(m2))
    np.save("b2.npy", sess.run(b2))

