#!/usr/bin/env python3
#This code reads the csv file, saves into TF graph to visualize, do clustering then save np array into .npy file
import csv
import numpy as np
def read_data(inpath = "train.csv"):
    with open(inpath) as csvfile:
        raw = csv.reader(csvfile)
        rawlist = []
        for row in raw:
            rawlist.append(row)
        #Remove first string row and convert everything into float32
        npdata = np.array(rawlist[1:], dtype = np.float32)
    #Remove first index column
    npdata = npdata[:, 1:]
    return npdata;
import tensorflow as tf
def save_data_ckpt(npdata, graphpath = "data.ckpt", cutfirst = False):
    embedding_var = tf.Variable(npdata)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, graphpath)

from sklearn.cluster import DBSCAN
def dbscan(npdata, outpath = "labels.tsv", eps = 0.5):
    clustering = DBSCAN(eps).fit_predict(npdata)
    print(clustering)
    length = len(set(clustering))
    with open(outpath, "w") as f:
        for line in np.array(clustering):
            f.write(str(line) + "\n")
    for i in range(len(clustering)):
        if (clustering[i] == -1):
            clustering[i] = length-1;
    return clustering

def save_npy(npdata, outpath):
    np.save(outpath, npdata)
    return None
'''
Xtrain = npdata[:,:11]
Yregressiontrain = npdata[:,11:]
Yclustertrain = np.zeros((clustering.shape[0], len(set(clustering))))
for i in range(len(clustering)):
    Yclustertrain[i, clustering[i]] = 1
print(Xtrain.shape)
print(Yregressiontrain.shape)
print(Yclustertrain.shape)
np.save("Xtrain.npy", Xtrain)
np.save("Yregressiontrain.npy", Yregressiontrain)
np.save("Yclustertrain.npy", Yclustertrain)
'''
