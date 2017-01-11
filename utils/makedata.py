#encoding=utf-8
'''Makedata有2种可能性：
1) 1个句子 len, w0, w1 ,...., w_(len-1), -1, ...
2) 一个center, 一组context w0,w1,...wk
第1种是数据量会=总的数据，第二的数据会是正常的K倍。
第2种在训练的时候更简单
'''
import resource
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

# make hdf5 work for fuel/blocks
def create_hdf5(sources, split, path_to):
    f = h5py.File(path_to, mode='w')
    for name in sources:
        f[name]=sources[name]
    split_dict = {}
    start=0
    for split_name,end in split:
        slice_=(start,end)
        split_dict[split_name]={name:slice_ for name in sources}
'''
# example of process
def process():
    vocab=get_vocab('vec.txt')
    p=IndexSequenceTransformer(sen_len=25,words=vocab)
    def indexmaker(texts=None):
        if not texts:
            return {'X':(None, 26)}
        else:
            X,y=p.transform(texts,return_sents=False)
            return {'X':X}
    return indexmaker 
'''
# maked chuncked hdf5 work for fuel/blocks
def chuncked_create_hdf5(process,input_path , output_path , 
        chuncked_batch_size=10000, batch_size=100, split={'train':[0,0.9],'develop':[0.9,1]}):
    '''
    process is function: 
            input None, return {'name1':shape, name2:shape2.。。}
            input texts, return {name1: data1, name2:data2}
    # sources should write chuncked
    # split should final computed by count and raw_split
    TODO: But how to make it random? use fuel iterface
    '''
    f = h5py.File(output_path, mode='w')
    for name,shape in process().iteritems():
        print name,shape
        f.create_dataset(name, (0, shape[1]), maxshape=(None, shape[1]), dtype='int32')
    def textss(path,n):
        ii=open(path,'r')
        if not n:yield ii.readlines()
        if n:
            while True:
                ss=[]
                for i in range(n):
                    s=ii.readline()
                    if s!='':
                        ss.append(s)
                    else: break
                if ss:
                    yield ss
                else:
                    break
    start=0
    for texts in textss(input_path,chuncked_batch_size):
        end=start+len(texts)
        data_dict=process(texts)
        #TODO process should be possible multi-processes
        print (start,end)
        for name in data_dict:
            print data_dict[name].shape
            f[name].resize(end,0) 
            f[name][start:end]=data_dict[name]
        start=end
    def ff(r):return int(r*end/batch_size)*batch_size 
    split_dict={a:{name:[ff(b[0]),ff(b[1])] for name in process()} for a,b in split.iteritems()}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        
if __name__=='__main__':
    # chuncked_create_hdf5(indexmaker, '../../../data/data/make_label/unlabel_pred.data/pre_data', 'data.hdf5')
    chuncked_create_hdf5(indexmaker, 'un.data', 'medium_data.hdf5')
    '''
    X=open('../../data/data/make_label/unlabel_pred.data/pre_data','r').readlines()
    X,y=IndexSequenceTransformer(sen_len=25,words=vocab).transform(X)
    N=X.shape[0]
    create_hdf5(sources={'X':X}, split=[('train',int(N*0.9/100)*100),('test',int(N/100)*100)], path_to='data.hdf5')
    '''
