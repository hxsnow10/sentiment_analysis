# encoding=utf-8
'''
simple sentiment analyisi based dict and rules
'''
from rule import MulRule
#dict based
from functools import partial
import numpy
import json
from nlp import segmentor, postagger, dpparser
from rule import *

class Dict_rule_senti():

    def __init__(self, senti_dict_path, rule_path=None, ltp_model_path=None):
        self.seg=segmentor
        ii=open(senti_dict_path,'r')
        a=ii.readlines()
        self.senti_dict={}
        for i in a:
            w,p=i.strip().split()
            if w[0] not in self.senti_dict:
                self.senti_dict[w[0]]=[]
            self.senti_dict[w[0]].append((w,p))
        if rule_path!=None:
            self.rule=MulRule(rule_path=rule_path, ltp_model_path=ltp_model_path)

    def senti_analysis(self, sent, pre=True):
        if pre:
            sent=' '.join(self.seg.transform(sent.strip()))
        words=sent.strip().split(' ')
        print 'words',' '.join(words)
        pp,nn=[],[]
        def senti(i):
            d=self.senti_dict
            rval=[]
            w=words[i]
            if w[0] not in d:return rval
            for ww,p in d[w[0]]:
                if p not in ['-1','1','0']:continue
                s=''
                for j  in range(i,len(words)):
                    s=s+words[j]
                    if len(s)>len(ww): break
                    if s==ww:
                        print 'dict pattern',' '.join(words[i:j+1]),ww,p
                        rval.append([(i,j+1),float(p)])
                        break
            return rval
        sentis=map(senti,range(len(words)))
        sentis=reduce(lambda x,y:x+y, sentis, [])
        print 'sentis',sentis
        label=[1,]*len(sentis)
        for i in range(len(sentis)):
            for j in range(len(sentis)):
                a=sentis[i]
                b=sentis[j]
                r0=a[0]
                r1=b[0]
                if r0[0]>=r1[0] and r0[1]<=r1[1] and a[1]!=b[1]:
                    label[i]=0
                    break
        sentis=[sentis[i] for i in range(len(sentis)) if label[i]==1]
        mul=self.rule.transform(words)
        for i in range(len(sentis)):
            r,t=sentis[i]
            m=mul[r[0]:r[1]]
            print m,numpy.all(m==numpy.mean(m))
            if numpy.all(m==numpy.mean(m)):
                sentis[i][1]*=numpy.mean(m)   
        pos_senti=[sentis[i] for i in range(len(sentis)) if sentis[i][1]>0]
        neg_senti=[sentis[i] for i in range(len(sentis)) if sentis[i][1]<0]
        #if len(neg_senti)*1.0/(len(pos_senti)+0.0001)>0.4 or len(neg_senti)>=2: return 0.1,None
        print 'sentis',len(sentis),sentis
        r=[sentis[i][1] for i in range(len(sentis)) if sentis[i][1]!=0]
        print r
        score=0.5*sum(r)/(len(r)+0.001)+0.5
        if score<=0.01:score=0.1
        if score>=0.99:score=0.9

        # score=0.5*merge(sentis)+0.5
        return score,None

def merge(a):
    a=[float(i) for i in a]
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

if __name__=='__main__':
    d=Dict_rule_senti(senti_dict_path='senti_dict.txt', rule_path='rules.cfg', ltp_model_path='ltp/ltp_data/')
    while True:
        sent=raw_input('Text=')
        print d.senti_analysis(sent)
