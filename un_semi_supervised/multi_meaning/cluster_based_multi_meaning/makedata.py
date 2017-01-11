#encoding=utf-8
'''Makedata有2种可能性：
1) 1个句子 len, w0, w1 ,...., w_(len-1), -1, ...
2) 一个center, 一组context w0,w1,...wk
第1种是数据量会=总的数据，第二的数据会是正常的K倍。
第2种在训练的时候更简单
'''
from utils.makedata import *

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
 
if __name__=='__main__':
    # chuncked_create_hdf5(indexmaker, '../../../data/data/make_label/unlabel_pred.data/pre_data', 'data.hdf5')
    chuncked_create_hdf5(process(), 'un.data', 'medium_data.hdf5')
    '''
    X=open('../../data/data/make_label/unlabel_pred.data/pre_data','r').readlines()
    X,y=IndexSequenceTransformer(sen_len=25,words=vocab).transform(X)
    N=X.shape[0]
    create_hdf5(sources={'X':X}, split=[('train',int(N*0.9/100)*100),('test',int(N/100)*100)], path_to='data.hdf5')
    '''
