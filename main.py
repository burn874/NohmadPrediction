data_offset = "data_file"
graph_offset = "tfgraph"
from python.data import read_data, dbscan, save_data_ckpt
import os
import pdb
import numpy as np
datapath = os.path.join(data_offset, "train.csv");
data = read_data(datapath);


clusterlabelpath = os.path.join(data_offset, "clusterlabels.tsv")
cluster = dbscan(data, clusterlabelpath);
print(set(cluster))
traindatagraphpath = os.path.join(graph_offset, "traindata.ckpt")
save_data_ckpt(data, traindatagraphpath)

from python.traincluster import train

train(data[:,0:11], cluster, os.path.join(graph_offset, "traingraph.ckpt"))

from python.testcluster import test

argmax = test(data[:,0:11], cluster, os.path.join(graph_offset, "traingraph.ckpt"))

print("Error {}/{}".format(np.count_nonzero(argmax - cluster), len(cluster)))



pdb.set_trace();





