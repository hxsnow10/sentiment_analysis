# encoding=utf-8
from cluster_theano import * 
from blocks.serialization import dump, load, load_parameters
from utils.makedata import *

class Multi_sense_model():

    def __init__(self, params_path, vec_path, sen_len=25):
        m=train(vec_path='vec.txt', batch_size=100, return_model=True)
        self.vocab=vocab=get_vocab(vec_path)
        params=load_parameters(open(params_path,'r'))
        m.set_parameter_values(params)
        self.clf=theano.function(m.inputs,m.outputs[0])
        self.ext=IndexSequenceTransformer(sen_len=sen_len,words=vocab)

    def build(self, index_X, meanings):
        '''build new X for retrain word2vec
        '''
        rval=[]
        for i in range(len(index_X)):
            x=index_X[i]
            y=meanings[i]
            r=[]
            print x[0],y[0]
            assert x[0]==y[0]
            for j in range(1,x[0]+1):
                nw=self.vocab[x[j]]+'_'+str(y[j])
                r.append(nw)
            rval.append(' '.join(r))
        return rval
    
    def transform(self, sents, return_newsents, input_path=None, output_path=None):
        #if not input_path:
            
        X=sents
        if X=='':y=[]
        else:
            X_,_=self.ext.transform(X,return_sents=True)
            y=self.clf(X_)
        if return_newsents:
            return y, self.build(X_,y)
        return y
         
    def batch_transform(self, sents, return_newsents, input_path=None, output_path=None):
        # TODO: truncted read and process
        batch_szie=100
        X=sents
        rval=[]
        for i in range(len(sents)/batch_size):
            X=sents[i*batch_szie:min((i+1)*batch_szie,len(sents))]
            X_,_=self.ext.transform([X])
            y=self.clf(X_)
            rval=rval+y
        y=rval
        if return_newsents:
            return y, self.build(X_,y)
        return y

def main():
    m=Multi_sense_model('cluster.pkl', 'vec.txt')
    sents=open('un.data2','r').readlines()
    y,X=m.transform(sents,return_newsents=True)
    oo=open('new_un_data2','w')
    for i in X:
        oo.write(i+'\n')
    oo.close()

if __name__=='__main__':
    main()
        
