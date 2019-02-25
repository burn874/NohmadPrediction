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
traindatagraphpath = os.path.join(graph_offset, "traindata.ckpt")
save_data_ckpt(data, traindatagraphpath)





pdb.set_trace();





