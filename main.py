data_offset = "data_file"
graph_offset = "tfgraph"
from python.data import read_data, clustering, save_data_ckpt
import os
import pdb
import numpy as np
datapath = os.path.join(data_offset, "train.csv");
data = read_data(datapath);

clusterlabelpath = os.path.join(data_offset, "clusterlabels.tsv")
cluster = clustering(data, clusterlabelpath, 14);
print(set(cluster))
traindatagraphpath = os.path.join(graph_offset, "traindata.ckpt")
save_data_ckpt(data, traindatagraphpath)

from python.traincluster import train

train(data[:,0:11], cluster, os.path.join(graph_offset, "traingraph.ckpt"))

from python.testcluster import test

argmax = test(data[:,0:11], os.path.join(graph_offset, "traingraph.ckpt"))

print("Error {}/{}".format(np.count_nonzero(argmax - cluster), len(cluster)))

from python.linearregression import linearregression

print("All data size {} correlation {}".format(data.shape[0], linearregression(data[:,0:11], data[:,11:13]).score(data[:,0:11], data[:,11:13])))

eff = 0

reglist = []

for i in set(argmax):
    subdatalist = []
    for j in range(data.shape[0]):
        if (argmax[j] == i):
            subdatalist.append(data[j])
    subdata = np.array(subdatalist, dtype = np.float32);
    reg = linearregression(subdata[:,0:11], subdata[:,11:13])
    reglist.append(reg)
    print("Cluster {} size {} correlation {}".format(i, subdata.shape[0], reg.score(subdata[:,0:11], subdata[:,11:13])))

    eff += subdata.shape[0] * reg.score(subdata[:,0:11], subdata[:,11:13]) / data.shape[0]

print("Eff correlation {}".format(eff))

testpath = os.path.join(data_offset, "test.csv")
testdata = read_data(testpath);

testdata = np.concatenate((testdata, np.zeros((1800, 11))), axis = 0)

argmax = test(testdata, os.path.join(graph_offset, "traingraph.ckpt"))

out = np.zeros([600, 2])

for i in range(600):
    out[i,:] = reglist[argmax[i]].predict(testdata[i].reshape([1, 11]));

with open("submit.csv", "w") as f:
    f.write("id,formation_energy_ev_natom,bandgap_energy_ev\n");
    for i in range(600):
        f.write(str(i+1)+","+str(out[i,0])+","+str(out[i,1])+"\n");


pdb.set_trace()




















