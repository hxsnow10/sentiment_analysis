# encoding=utf-8

import os
os.environ['THEANO_FLAGS']="device=cpu"

from numpy import savez,savez_compressed,load
np_load=load
from blocks.serialization import dump, load, load_parameters
from cluster_theano import *

def get_vocab(vocab_path):
    ii=open(vocab_path,'r')
    ii.readline()
    vocab=[]
    for i in ii: 
        vocab.append(i.strip().split()[0])
    return vocab

def get_multi_vector_result(params_path, vec_path):
    params=load_parameters(open(params_path,'r'))
    print params.keys()

    multi_vector=params['/kmeans.WordCentroids']
    prototype_nums=multi_vector.shape[1]
    
    oo=open('multi_vector.txt','w')
    vocab=get_vocab(vec_path)
    for i in range(len(vocab)):
        for j in range(prototype_nums):
            new_name=vocab[i]+'_'+str(j)
            oo.write(new_name+' '+' '.join([str(x) for x  in list(multi_vector[i][j])])+'\n')
    oo.close()

if __name__=='__main__':
    get_multi_vector_result('cluster.pkl','vec.txt')

