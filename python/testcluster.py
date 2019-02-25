#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import pdb

def test(Xtrain, Ycluster, traingraphpath):
    saver = tf.train.import_meta_graph(traingraphpath + ".meta")
    with tf.Session() as sess:
        saver.restore(sess, traingraphpath)
        X = sess.graph.get_tensor_by_name("Xtrain:0")
        h = sess.graph.get_tensor_by_name("softmax:0")
        prediction = sess.run(h, feed_dict = {X:Xtrain})
        argmax = np.argmax(prediction, axis = 1);

        return argmax;
