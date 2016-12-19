# encoding=utf-8

# fenci
# preprocessing
# train wordvec from big data
# transfrom data
# make hdf5

from fuel.datasets.hdf5 import H5PYDataset
import h5py
from numpy import load
import numpy
import numpy as np

# make hdf5
def create_hdf5(sources, split, path_to):
    f = h5py.File(path_to, mode='w')
    for name in sources:
        f[name]=sources[name]
    split_dict = {}
    start=0
    for split_name,end in split:
        slice_=(start,end)
        split_dict[split_name]={name:slice_ for name in sources}    
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

if __name__=='__main__':
    train_X = numpy.load('train_X.npy')
    test_X = numpy.load('test_X.npy')
    train_y = numpy.load('train_y.npy')
    test_y = numpy.load('test_y.npy')
    X=numpy.vstack([train_X,test_X])
    y=numpy.vstack([train_y,test_y])
    '''
    X.dims[0].label = 'batch'
    X.dims[1].label = 'index'
    y.dims[0].label = 'batch'
    y.dims[1].label = 'index'
    '''
    N=X.shape[0]
    print N,N*0.9
    create_hdf5(sources={'X':X,'y':y}, split=[('train',int(N*0.9/100)*100),('test',int(N/100)*100)], path_to='data.hdf5')
