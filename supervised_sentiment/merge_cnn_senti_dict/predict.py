from cnn_model import *
from numpy import savez,savez_compressed,load
np_load=load

def load_cnn(vocab_path='vocab.txt',params_path='params.npz',batch_size=100,sen_len=50)
    model=main(vocab_path=vocab_path, batch_size=batch_szie, return_model=True, sen_len=sen_len)
    param_dict=model.get_parameter_values()

    params=np_load(params_path)
    d={}
    for key in params:
        d[('/'+key).replace('convolutionalsequence','feedforwardsequence')]=params[key]
    left=0
    for i in range(5):
        right=left+param_dict['/sentinet/multiconv/feedforwardsequence/linear_{}/linear_0.W'.format(i)].shape[0]
        d['/sentinet/multiconv/feedforwardsequence/linear_{}/linear_0.b'.format(i)]=params['sentinet/mlp/linear_0.b']
        d['/sentinet/multiconv/feedforwardsequence/linear_{}/linear_0.W'.format(i)]=params['sentinet/mlp/linear_0.W'][left:right,:]
        
    model.set_parameter_values(d)
    return model

