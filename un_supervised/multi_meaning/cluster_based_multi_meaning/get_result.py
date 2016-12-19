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
'''
main_loop=load(open('cluster.pkl','r'))
w=open('params.npz','w')
model=main_loop.model
param_dict=model.get_parameter_values()
print param_dict.keys()
savez(w, **param_dict)
'''
params_dict=np_load('params.npz')
print params_dict.keys()
multi_vector=params_dict['kmeans.WordCentroids']

prototype_nums=3
oo=open('multi_vector.txt','w')
vocab=get_vocab('vec2.txt')
for i in range(len(vocab)):
    for j in range(prototype_nums):
        new_name=vocab[i]+'_'+str(j)
        oo.write(new_name+' '+' '.join([str(x) for x  in list(multi_vector[i][j])])+'\n')
oo.close()
