# encoding=utf-8
import os
import numpy as np
from textclf.data import *
import h5py
from fuel.datasets.hdf5 import H5PYDataset

def get_vocab(vocab_path):
    ii=open(vocab_path,'r')
    vocab=[]
    for i in ii: 
        vocab.append(i.strip().split()[0])
    return vocab

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
    raw_data_path='test'
    vocab_path='vocab.txt'
    cmd='nohup word2vec -train '+raw_data_path+' -min-count 5 -save-vocab vocab.txt -iter 0'
    os.system(cmd)
    window=7
    half=(window-1)/2
    vocab=get_vocab('vocab.txt')
    word2index={}
    for k,w in enumerate(vocab):
        word2index[w]=k
    print len(vocab)
    ii=open(raw_data_path,'r')
    con=[]
    cen=[]
    for i in ii:
        # should i cut sentence
        words=[j for j in i.strip().split() if j in vocab]
        print len(words)
        words=['</s>',]*half+words+['</s>',]*half
        for k in range(half,len(words)-half):
            central=[word2index[words[k]]]
            left_context=[word2index[w] for w in words[k-half:k]]
            right_context=[word2index[w] for w in words[k+1:k+half+1]]
            context=left_context+right_context
            con.append(context)
            cen.append(central)
    con=np.array(con)
    cen=np.array(cen)
    
    n=len(con)
    print n
    train_con = con[0:int(n*0.9)]
    test_con = con[int(n*0.9):n]
    train_cen = cen[0:int(n*0.9)]
    test_cen = cen[int(n*0.9):n]
    print train_cen.shape,test_cen.shape
    X=numpy.vstack([train_con,test_con])
    y=numpy.vstack([train_cen,test_cen])
    N=X.shape[0]
    print N,N*0.9
    create_hdf5(sources={'con':X,'cen':y}, split=[('train',int(N*0.9/100)*100),('test',int(N/100)*100)], path_to='data.hdf5')
