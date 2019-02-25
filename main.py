data_offset = "data_file"
from python.data import read_data, dbscan
import os
import pdb
datapath = os.path.join(data_offset, "train.csv");
data = read_data(datapath);
clusterlabelpath = os.path.join(data_offset, "clusterlabels.tsv")
cluster = dbscan(data, clusterlabelpath);
pdb.set_trace();





