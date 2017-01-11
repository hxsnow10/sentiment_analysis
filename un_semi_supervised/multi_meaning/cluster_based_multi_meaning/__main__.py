from cluster_theano import train
from get_result import get_multi_vector_result 

if __name__ == "__main__":
    #TODO: change theano variable
    #TODO: makedata
    train(data_path='small_data.hdf5')
    get_multi_vector_result('cluster.pkl','vec.txt')
    
