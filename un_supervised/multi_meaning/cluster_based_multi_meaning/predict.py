# encoding=utf-8
from cluster_theano import * 

class Multi_sense_model():

    def __init__(self, params_path):
        m=main(vocab_path='vocab2.txt', batch_size=100, return_model=True)
        params=np_load(params_path)
        d={}
        for key in params:
            d['/'+key]=params[key]
        m.set_parameter_values(d)
        self.clf=theano.function(m.inputs,m.outputs[0])
        self.ext=IndexSequenceTransformer(sen_len=sen_len,words=vocab)

    def build(self, index_X, meanings):
        rval=[]
        for i in range(index_X.shape[0]):
            x=index_X[i]
            y=meanings[i]
            r=[]
            assert x[0]==y[0]
            for j in range(1,x[0]+1):
                nw=self.vocab[x[j]]+'_'+str(y[j])
                r.append(nw)
            rval.append(' '.join(r))
        return rval
    
    def transform(self, sent, return_newsents):
        X=sent
        if X=='':y=[]
        else:
            X_,_=self.ext.transform([X],return_sents=True)
            y=self.clf(X_)
        if return_newsents:
            return y[0], build(X,y)[0]
        return y
         
    def batch_transform(self, sents, return_newsents):
        batch_szie=100
        rval=[]
        for i in range(len(sents)/batch_size):
            X=sents[i*batch_szie:min((i+1)*batch_szie,len(sents))]
            X_,_,new_sents=self.ext.transform([X])
            y=self.clf(X_)
            rval=rval+y
        if return_newsents:
            return y, build(X,y)
        return y

                

if __name__=='__main__':
    m=Multi_sense_model('params.npz')
        
