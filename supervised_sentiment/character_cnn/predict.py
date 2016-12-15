# encoding=utf-8
import readline 
from pre import Preprocess
from pyltp import Segmentor, Postagger, Parser
from ccnn_model import *
from textclf.model.ext import IndexSequenceTransformer 
import pickle
from blocks.serialization import load
from numpy import load as np_load
from simple_senti import *
class allModel():

    def __init__(self, vocab_path='chara.txt', sen_len=100, params_path='params.npz', senti_dict_path=None, rules_path=None,
            nn_path=None, pos_path='./ltp/ltp_data/pos.model'):
        vocab=self.get_vocab(vocab_path,'\n')
        self.pre=Preprocess()
        self.postagger = Postagger() # 初始化实例
        self.postagger.load(pos_path)  # 加载模型
        self.ext=IndexSequenceTransformer(sen_len=100,words=vocab)
        m=main(vocab_path=vocab_path, batch_size=100, return_model=True)
        params=np_load(params_path)
        d={}
        for key in params:
            d['/'+key]=params[key]
        m.set_parameter_values(d)
        self.clf=theano.function(m.inputs,m.outputs[0])
        self.dict_senti=None
        if senti_dict_path!=None:
            self.dict_senti=Dict_rule_senti(senti_dict_path, rules_path)
        self.nn=None
        if nn_path!=None:
            self.nn=set(self.get_vocab(nn_path,'\t'))
        
        
    def predict(self, X, pre=False):
        print X
        try:
            if pre:
                X_=self.pre.transform(X)[0]
            else:X_=X
            words=X_.split()
            #poss=self.postagger.postag(words)
            #words=[i[0] for i in zip(words,poss) if not (i[1][0]=='n' and i[0] not in self.nn)]

            X_=' '.join(words)
            X_='\t'.join([i.encode('utf-8') for i in X_.decode('utf-8')])
            print 'xxxxxxxxxxxxxxxxxxxxxx',X_
            X_,_=self.ext.transform([X_],split='\t')
            y=self.clf(X_)[0]
            print y
            #if self.dict_senti!=None:
            #    y2=self.dict_senti.senti_analysis(X,pre=pre)
            #    print y2
            #    y=min(y,y2[0])
            print y
            #return float(y)
        except Exception,e:
            print e
            return 0.5

    def get_vocab(self, vocab_path, split=' '):
        ii=open(vocab_path,'r')
        vocab=[]
        for i in ii:
            vocab.append(i.strip().split(split)[0])
        return vocab

if __name__=='__main__':
    m=allModel(vocab_path='chara.txt',params_path='params.npz', senti_dict_path='senti_dict.txt', rules_path='rules.cfg', nn_path='nn.txt')
    while True:
        sent=raw_input('Enter Sent:')
        print m.predict(sent,pre=True)
        # print m.predict(sent,pre=False)
