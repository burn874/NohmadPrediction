#!/usr/bin/env python3
#This file reads .npy file (X, Ycluster) then uses 3 layers neural net to try to train a classifier.
import numpy as np
import tensorflow as tf

def train(Xtrain, Ycluster, traingraphpath):
    num_samples = Xtrain.shape[0];
    inshape = Xtrain.shape[1]
    intermediate = 6*inshape
    outshape = len(set(Ycluster))

    Yclustertrain = np.zeros(shape = (num_samples, outshape))
    for i in range(num_samples):
        if (Ycluster[i] == -1):
            Yclustertrain[i][outshape-1] = 1
        else:
            Yclustertrain[i][Ycluster[i]] = 1

    print(Xtrain.shape)
    print(Yclustertrain.shape)
    X = tf.placeholder(dtype = tf.float32, shape = Xtrain.shape, name="Xtrain")
    Y = tf.placeholder(dtype = tf.float32, shape = Yclustertrain.shape, name = "Yclustertrain")
    #h = softmax(relu(X*m1 + b1)*m2 + b2)
    m1 = tf.Variable(tf.random.normal((inshape, intermediate), dtype = tf.float32, stddev=0.01), name = "m1")
    b1 = tf.Variable(tf.random.normal((1, intermediate), dtype = tf.float32, stddev=0.01), name = "b1")
    m2 = tf.Variable(tf.random.normal((intermediate, outshape), dtype = tf.float32, stddev=0.01), name = "m2")
    b2 = tf.Variable(tf.random.normal((1, outshape), dtype = tf.float32, stddev=0.01), name = "b2")

    h = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(X, m1) + b1), m2) + b2, axis = 1, name = "softmax")
    print(h)
    loss = tf.reduce_mean(-(Y*tf.log(h) + (1-Y)*tf.log(1-h)), name = "loss")
    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
    trainer = optimizer.minimize(loss, name = "trainer")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(600):
            _, lossvalue = sess.run([trainer, loss], feed_dict = {X:Xtrain, Y:Yclustertrain})
            print(lossvalue)
#        print(sess.graph.get_operations())
        saver.save(sess, traingraphpath)
        print("Graph is saved into {}".format(traingraphpath))
