# encoding=utf-8
from cluster_theano import *
from numpy import savez,savez_compressed,load
np_load=load
from blocks.serialization import dump, load, load_parameters

import os
os.environ['THEANO_FLAGS']="device=cpu"

main_loop=load(open('cnn.pkl','r'))
w=open('params.npz','w')
model=main_loop.model
param_dict=model.get_parameter_values()
print param_dict.keys()
savez(w, **param_dict)

w.close()
x = tensor.lmatrix('X')
m=main(vocab_path='vocab.txt', batch_size=100, return_model=True)
params=np_load('params.npz')
d={}
for key in params:
    d['/'+key]=params[key]
m.set_parameter_values(d)
# m.set_parameter_values(params) 
# m.set_parameter_values(parameters)
# m=senti_net()


if __name__=='__main__':
    m=main(vocab_path='vocab2.txt', batch_size=100, return_model=True)
    print m.get_parameter_values().keys()
    # m.set_parameter_values(parameters)
    # m=senti_net()
    
