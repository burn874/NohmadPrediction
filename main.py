data_offset = "data_file"
graph_offset = "tfgraph"
from python.data import read_data, clustering, save_data_ckpt
import os
import pdb
import numpy as np
datapath = os.path.join(data_offset, "train.csv");
data = read_data(datapath);


clusterlabelpath = os.path.join(data_offset, "clusterlabels.tsv")
cluster = clustering(data, clusterlabelpath, 15);
print(set(cluster))
traindatagraphpath = os.path.join(graph_offset, "traindata.ckpt")
save_data_ckpt(data, traindatagraphpath)

from python.traincluster import train

train(data[:,0:11], cluster, os.path.join(graph_offset, "traingraph.ckpt"))

from python.testcluster import test

argmax = test(data[:,0:11], cluster, os.path.join(graph_offset, "traingraph.ckpt"))

print("Error {}/{}".format(np.count_nonzero(argmax - cluster), len(cluster)))

from python.linearregression import linearregression

print("Global data {} correlation {}".format(data.shape[0], linearregression(data[:,0:11], data[:,11:13])))

for i in set(argmax):
    subdatalist = []
    for j in range(data.shape[0]):
        if (argmax[j] == i):
            subdatalist.append(data[j])
    subdata = np.array(subdatalist, dtype = np.float32);
    print("Subdata {} correlation {}".format(subdata.shape[0], linearregression(subdata[:,0:11], subdata[:,11:13])))
























