#encoding=utf-8
from textclf.data import *
from textclf.model.packs import *
from textclf.vocab.words import Vocabulary
from fuel.datasets.hdf5 import H5PYDataset
import h5py
from numpy import load
import numpy
import numpy as np

def get_vocab(vocab_path):
    ii=open(vocab_path,'r')
    vocab=[]
    for i in ii:
        vocab.append(i.strip().split()[0])
    return vocab

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
    vocab=get_vocab('vocab.txt')
    X,y,_=raw().transform('data')
    X,y=IndexSequenceTransformer(sen_len=40,words=vocab).transform(X,y)#允许词典中有非极性词，在分词的时候把非极性词过滤
    Makedata(X,y,data_split=[('train',0.9),('test',1)])

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
