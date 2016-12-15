#encoding=utf-8
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer,TfidfTransformer
#__all__=[]

import sys
import re

import numpy as np
import jieba
from jieba import posseg
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class IndexSequenceTransformer(TransformerMixin):

    def __init__(self,sen_len=50,words=None):
        self.sen_len=sen_len
        self.word_to_index={}
        for i,w in enumerate(words):
            self.word_to_index[w]=i
        self.word_num=len(words)
 
    def transform(self,X,y=[],split=' '):
        '''
        return a numpy of sequence
        format x: 
            x[0]:num of index of read word
            x[1:]:indexs when index>=0, is really index, else -1 is not.
        '''
        y=self.one_hot(y)
        new_X=[0,]*len(X)
        sum_index,sum=0,0
        sen_len=self.sen_len
        for i,line in enumerate(X):
            words=line.strip().split(split)
            poss=self.postagger.tag(words)
            dp=self.dp.dp(words,poss)

            x=[]
            for w in words:
                if w in self.word_to_index:
                    x.append(self.word_to_index[w])
                if len(x)==self.sen_len:
                    break
            sum_index,sum=sum_index+len(x),sum+len(words)
            x_len=len(x)
            while(len(x)<sen_len):
                x.append(-1)
            x=[x_len]+x
            new_X[i]=np.array(x)
        print 'frac=',1.0*sum_index/sum
        return np.array(new_X,dtype=np.int64),np.array(y,dtype=np.int64)
    
    def one_hot(self,y):
        classes={}
        if y!=[]:
            for i in y:
                if i  not in classes:
                    print 'class ', i, 'added'
                    classes[i]=len(classes)
            Y=np.zeros([len(y),len(classes)])
            for k in range(len(y)):
                Y[k,classes[y[k]]]=1
            return Y
        return y
