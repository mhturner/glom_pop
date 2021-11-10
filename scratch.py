import os
from mat4py import loadmat
import numpy as np

dir = '/Users/mhturner/Downloads'

dr = loadmat(os.path.join(dir, 'dataMap.mat')).get('dataRoundup')

for k in range(len(dr.get('cellID'))):
    vals = list(dr.get('ContrastF1F2')[k].values())
    if np.sum(vals) > 0:
        # print(k)
        # print(dr.get('cellType')[k])
        print(dr.get('cellID')[k])



# %%
mat = scipy.io.loadmat(os.path.join(dir, 'dataMap.mat'))

type(mat)
mat.keys()
dr = mat.get('dataRoundup')
type(dr)
dr.shape
type(dr[0])

dr.shape
dr[0, 101]
tt = dr[0, 101]
tt.shape
tt[1]
len(tt)
print(dr[0].keys())
