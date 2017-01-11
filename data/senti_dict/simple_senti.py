# encoding=utf-8
'''
simple sentiment analyisi based dict and rules
'''
from textclf import pre

#dict based
class dict_senti():

    def __init__(path, pre='default'):
        a=open(path,'r').readlines()
        if pre='default':
            self.pre=Preprocess()
        self.senti_dict={}
        for i in a:
            w,p=i.split('\t')
            self.senti_dict[w]=p
        
    def senti_analysis(sent, need_fenci=True):
        if need_fenci:
            sent=self.pre.transform(sent)
        words=sent.split(' ')
        pos,neg=0,0
        for word in words:
            if word in senti_dict:
                if senti_dict[word]=='1':
                    pos+=1
                if senti_dict[word]=='-1':
                    neg+=1
        score=pos*1.0/(po+neg+0.1)
        return (pos,neg)
