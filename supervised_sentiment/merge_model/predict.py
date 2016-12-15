# encoding=utf-8
#import readline 
from pre import Preprocess
from pyltp import Segmentor, Postagger, Parser
from cnn_model import *
from index_transformer import IndexSequenceTransformer 
import pickle
from blocks.serialization import load
from numpy import load as np_load
from simple_senti import *
from merge_model import merge_table_senti 
def load_cnn(vocab_path='vocab.txt',params_path='params.npz',batch_size=100,sen_len=50):
    model=main(vocab_path=vocab_path, batch_size=batch_size, return_model=True, sen_len=sen_len)
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

class allModel():

    def __init__(self, vocab_path='vocab.txt', sen_len=50, params_path='params.npz', senti_dict_path=None, rules_path=None,
        nn_path=None, pos_path='./ltp/ltp_data/pos.model', black_path=None, user_dict_path='user_dict.txt'):
        vocab=self.get_vocab(vocab_path)
        self.pre=Preprocess(userdict_path=user_dict_path)
        self.postagger = Postagger() # 初始化实例
        self.postagger.load(pos_path)  # 加载模型
        self.ext=IndexSequenceTransformer(sen_len=50,words=vocab)
        m=load_cnn(vocab_path=vocab_path, params_path=params_path, batch_size=100, sen_len=50)
        self.clf=theano.function(m.inputs,m.outputs[0])
        self.dict_senti=None
        if senti_dict_path!=None:
            self.dict_senti=Dict_rule_senti(senti_dict_path, rules_path)
        self.nn=None
        if nn_path!=None:
            self.nn=set(self.get_vocab(nn_path))
            print len(self.nn)
        if black_path!=None:
            self.black=set(self.get_vocab(black_path))
        
    def predict(self, X, pre=False):
        print 'input',X
        if pre:
            X_=self.pre.transform(X)[0]
        else:X_=X
        words=X_.split()
        print 'pred','\t'.join(words),len(words)
        poss=self.postagger.postag(words)
        print list(poss)
        words=[i[0] for i in zip(words,poss) 
            if not ((i[1] in ['n','ws','x','u'])
        and i[0] not in self.nn) and not (i[0] in self.black)]
        # filter more 
        X_new=' '.join(words)
        print 'sentiment processed ',X_new
        y1=np.ones((5,len(words)))
        if X_new!='':
            X_,_,in_mask=self.ext.transform([X_new])
            y=self.clf(X_)[0]
            in_mask=in_mask[0]
            print y[:,:6]
            return 'end'
            for i in range(len(in_mask)):
                y1[:,i]=in_mask[i]*y[:,i]
            print 'cnn result',y1
        if self.dict_senti!=None:
            y2=self.dict_senti.senti_analysis(X_new,pre=False)
            print 'dict senti',y2
        y=merge_table_senti(y1,y2)
        print 'final result',y
        return y
        #return float(y)
    
    def merge(self, a):
        a=[float(i) for i in a]
        # print a
        if a==[]:return 0.5
        ans1=min(a)
        def weight(x):
            if abs(x-0.5)<0.1:return 0
            else:return (x-0.5)*(x-0.5)
        w=map(weight,a)
        if sum(w)==0:ans2=0.5
        else:
            ans2=sum([i*j for i,j in zip(a,w)])*1.0/(sum(w))
        return ans2
    
    def get_vocab(self, vocab_path):
        ii=open(vocab_path,'r')
        vocab=[]
        for i in ii:
            vocab.append(i.strip().split()[0])
        ii.close()
        return vocab

if __name__=='__main__':
    m=allModel(vocab_path='vocab.txt',params_path='params.npz', senti_dict_path='senti_dict.txt', rules_path='rules.cfg', nn_path='nn.txt', black_path='black.txt')
    while True:
        sent=raw_input('Enter Sent:')
        print m.predict(sent,pre=True)
        #print m.predict(sent,pre=False)
