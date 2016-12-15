# encoding=utf-8
'''
simple sentiment analyisi based dict and rules
'''
from rule import MulRule
from functools import partial
import numpy
import json
from nlp import segmentor, postagger, dpparser
from rule import *
from copy import deepcopy

class Dict_rule_senti():

    def __init__(self, senti_dict_path, rule_path=None, ltp_model_path=None):
        self.seg=segmentor
        ii=open(senti_dict_path,'r')
        a=ii.readlines()
        self.senti_dict={}
        for i in a:
            w,p=i.strip().split()
            if w[0] not in self.senti_dict:#TODO
                self.senti_dict[w[0]]=[]
            self.senti_dict[w[0]].append((w,p))
        if rule_path!=None:
            self.rule=MulRule(rule_path=rule_path, ltp_model_path=ltp_model_path)

    def senti_analysis(self, sent, pre=True):
        if sent.strip()=='':return 0.5,{},{}
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
                if r0[0]>=r1[0] and r0[1]<=r1[1] and r0!=r1:
                    label[i]=0
                    break
        sentis=[sentis[i] for i in range(len(sentis)) if label[i]==1]
        norule_sentis=deepcopy(sentis)
        mul=self.rule.transform(words)
        for i in range(len(sentis)):
            r,t=sentis[i]
            m=mul[r[0]:r[1]]
            print m,numpy.all(m==numpy.mean(m))
            if numpy.all(m==numpy.mean(m)):
                sentis[i][1]*=numpy.mean(m)   
        print sentis
        print 'xxxxxxxxxxxxxxxxxxxx',len(sentis) 
        r=[sentis[i][1] for i in range(len(sentis)) if sentis[i][1]!=0]
        print r
        score=0.5*sum(r)/(len(r)+0.001)+0.5
        if score<=0.01:score=0.1
        if score>=0.99:score=0.9
        norule_debug=sentiment_conflict_debug(sent, words,norule_sentis, sent_score=score)
        rule_debug=sentiment_conflict_debug(sent, words, sentis, sent_score=score)
        # score=0.5*merge(sentis)+0.5
        return score, norule_debug, rule_debug

def direct_distance(r1, r2):
    jiaocha=False
    if abs(min(r1[0],r2[0])-max(r1[1],r2[1]))-(r1[1]-r1[0])-(r2[1]-r2[0])>0:
        jiaocha=True
    if jiaocha:return 0
    else:return min(abs(r1[0]-r2[1]),abs(r2[0]-r1[1]))

def tree_distance(r1,r2,d):
    rval=1000
    for i in range(r1[0],r1[1]):
        for j in range(r2[0],r2[1]):
            rval=min(rval,d[i,j])
    return rval
    
def dist(words):
    '''return matrix of N*N, N=len(words)
    '''
    N=len(words)
    postags = postagger.transform(words)
    arcs = parser.transform(words, postags)
    arcs=[(arc.head-1, i) for i,arc in enumerate(arcs) if arc.head!=0]
    dist=np.zeros((N,N))*(2*N)
    for i,j in arcs:
        dist[i,j]=1
        dist[j,i]=1
        dist[i,i]=dist[j,j]=0
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i,j]>dist[i,k]+dist[k,j]:
                    dist[i,j]=dist[i,k]+dist[k,j]
    return dist 

def sentiment_conflict_debug(sent, words,  sentis, sent_score=None, max_direct_distance=3, max_tree_dictance=1):
    global_conflicts, local_conflicts=[], []
    if sent_score:
        global_conflicts=[]
        for r,p in sentis:
            if sent_score>0.7 and p<0 or sent_score<0.3 and p>0:
                global_conflicts.append((p,''.join(words[r[0]:r[1]])))
    for i in range(len(sentis)):
        for j in range(len(sentis)):
            if i==j: continue
            r1,p1 = sentis[i]
            r2,p2 = sentis[j]
            #if p1*p2<0 and (tree_distance(r1,r2,d)<=max_tree_dictance):
            if p1*p2<0 and (direct_distance(r1,r2)<=max_direct_distance or tree_distance(r1,r2)<=max_tree_dictance):
                local_conflicts.append([(p1,''.join(words[r1[0]:r1[1]])),(p2,''.join(words[r2[0]:r2[1]]))])
    rval={'sent':sent, 'global_conflicts':global_conflicts, 'local_conflicts':local_conflicts,'score':sent_score}
    return rval

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
