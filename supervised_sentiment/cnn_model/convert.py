# encoding=utf-8

from numpy import savez,savez_compressed,load
np_load=load
from blocks.serialization import dump, load, load_parameters
from cnn_model import *

if __name__=='__main__':
    m=main(vocab_path='vocab2.txt', batch_size=100, return_model=True)
    print m.get_parameter_values().keys()
    # m.set_parameter_values(parameters)
    # m=senti_net()
    
