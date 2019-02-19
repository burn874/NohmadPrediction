#!/usr/bin/env python
# coding: utf-8

#This code reads the csv file, saves into TF graph to visualize, do clustering then save np array into .npy file





# In[1]:


import csv
import numpy as np


# In[6]:


with open("train.csv") as csvfile:
    raw = csv.reader(csvfile)
    rawlist = []
    for row in raw:
        rawlist.append(row)
    npdata = np.array(rawlist[1:], dtype = np.float32)
npdata = npdata[:, 1:]
print("Shape {}".format(npdata.shape))


# In[8]:


import tensorflow as tf
embedding_var = tf.Variable(npdata)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "./data.ckpt")


# In[22]:


#TRY TO SPLIT DATA INTO 11 CLUSTERS
a = np.sum((npdata[2371] - npdata[2399])**2)
print("Minimum euclid distance: {}".format(a))
from sklearn.cluster import DBSCAN
clustering = DBSCAN().fit_predict(npdata)
print(clustering)
print(set(clustering))
list(clustering).index(-1)
with open("labels.tsv", "w") as f:
    for line in np.array(clustering):
        f.write(str(line) + "\n")


# In[ ]:


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


# In[ ]:




