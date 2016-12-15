# encoding=utf-8
import os
os.environ['THEANO_FLAGS']="device=cpu"

from numpy import savez,savez_compressed,load
np_load=load
from blocks.serialization import dump, load, load_parameters
from cnn_merge import *

main_loop=load(open('cnn.pkl','r'))
w=open('params.npz','w')
model=main_loop.model
param_dict=model.get_parameter_values()
savez(w, **param_dict)


w.close()
x = tensor.lmatrix('X')
m=main(vocab_path='vocab.txt', batch_size=100, return_model=True)
print m.get_parameter_dict()
params=np_load('params.npz')
d={}
for key in params:
    d['/'+key]=params[key]
    print key,params[key].shape
m.set_parameter_values(d)
# m.set_parameter_values(params) 
# m.set_parameter_values(parameters)
# m=senti_net()

