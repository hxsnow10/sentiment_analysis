# encoding=utf-8
import readline 
from pre import Preprocess
from cnn_model import *
from textclf.model.ext import IndexSequenceTransformer 
import pickle
from blocks.serialization import load
from numpy import load as np_load
from simple_senti import *

class allModel():

    def __init__(self, vocab_path='vocab.txt', sen_len=40, params_path='params.npz', senti_dict=True):
        vocab=self.get_vocab(vocab_path)
        self.pre=Preprocess()
        self.ext=IndexSequenceTransformer(sen_len=sen_len,words=vocab)
        m=main(vocab_path=vocab_path, batch_size=100, return_model=True, sen_len=sen_len)
        params=np_load(params_path)
        d={}
        for key in params:
            d['/'+key]=params[key]
        m.set_parameter_values(d)
        self.clf=theano.function(m.inputs,m.outputs[0])
        self.dict_senti=None
        if senti_dict!=None:
            self.dict_senti=Dict_rule_senti(senti_dict_path='senti_dict.txt', rule_path='rules.cfg')

        
    def predict(self, X_):
        try:
            X=self.pre.transform(X_)
            X,_=self.ext.transform(X)
            y=self.clf(X)[0][1]
            '''
            if self.dict_senti!=None:
                y2=self.dict_senti.senti_analysis(X_)[0]
                y=min(y,y2)
            '''
            print y
            return float(y)
        except Exception,e:
            print e
            return 0.5

    def get_vocab(self, vocab_path):
        ii=open(vocab_path,'r')
        vocab=[]
        for i in ii:
            vocab.append(i.strip().split()[0])
        return vocab

if __name__=='__main__':
    m=allModel(vocab_path='vocab.txt',params_path='params.npz', )
    while True:
        sent=raw_input('Enter Sent:')
        print m.predict(sent)
