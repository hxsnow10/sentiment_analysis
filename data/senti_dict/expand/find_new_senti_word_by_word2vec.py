# encoding=utf-8
from gensim.models import word2vec
import numpy
from copy import deepcopy
from numpy import ndarray
import time
from multiprocessing import Pool

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# read seed senti dict
ii=open('senti_dict.txt','r')
d={}
for i in ii:
    w,p=i.strip().split()
    if p not in ['0','1','-1']:p='0'
    d[w.decode('utf-8')]=p
ii.close()

# read word vectors
start=time.time()
model=word2vec.Word2Vec.load_word2vec_format('vec.txt',binary=False)
print time.time()-start

new_d={w:[numpy.array([0,0,0]),{}] for w in model.vocab}

# build top similar first
'''
most_similar={}
words=model.vocab.keys()
def f(x):
    return model.most_similar(positive=[x])
pool=Pool(20)
sims=pool.map(f,words)
for w,sim in zip(words,sims):
    most_similar[w]=sim
'''

print time.time()-start
def get_score(label):
    if label=='0':
        return numpy.array([0,1,0]),{w:numpy.array([0,1,0])}
    elif label=='1':
        return numpy.array([0,0,1]),{w:numpy.array([0,0,1])}
    elif label=='-1':
        return numpy.array([1,0,0]),{w:numpy.array([1,0,0])}
    else:
        return numpy.array([0,0,0]),{}

def merge(x,y):
    z=deepcopy(x)
    for k,v in y.iteritems():
        if k not in z:
            z[k]=v
        else:
            z[k]+=v
    return z

for w in d:
    if w in new_d:
        p,s=get_score(d[w])
        new_d[w]=[p,s]
r=1
k=1

# 传播来提升每个词的情感分布
while k>0:
    new_dd=deepcopy(new_d)
    r=r*0.8
    for w in new_d:
        if  (new_d[w][0]!=numpy.array([0,0,0])).any():
            similar=model.most_similar(w, topn=200)
            for ww,p in similar:
                if p>=0.5:
                # print ww,new_dd[ww]
                # print p*r*new_d[w] 
                    scale=p*p*r
                    new_dd[ww][0]=new_dd[ww][0]+p*p*r*new_d[w][0]
                    new_dd[ww][1]=merge(new_dd[ww][1], {k:scale*v for k,v in new_d[w][1].iteritems()})
    new_d=new_dd
    k-=1

# give result
result=[[],[],[]]
conflict=[]
for w in new_d:
    p,s=new_d[w]
    if (p!=numpy.array([0,0,0])).any():
        p=p/(p.sum()+0.0001)
        index=ndarray.argmax(p)
        #supports1=[(ww,vv) for ww,vv in s.iteritems() if ndarray.argmax(vv)==index]
        supports2=[ww.encode('utf-8') for ww,vv in s.iteritems() if ndarray.argmax(vv)==index]
        supports2=','.join(supports2)
        probs=p[index]
        if w in d: 
            if str(index-1)!=d[w]:
                conflict.append((w.encode('utf-8'), d[w], str(index-1), str(probs), supports2))
                
        else: result[index].append((w.encode('utf-8'),str(probs),supports2))

# print conflict word 
oo=open('conflict.txt','w')
oo.write('\t'.join(['word','orginal','new','probs'])+'\t')
conflict=sorted(conflict, key=lambda x:(x[2],float(x[3])),reverse=True)
for i in conflict:
    oo.write('\t'.join(i)+'\n')
oo.close()

# print senti result
print [len(i) for i in result]
for i in range(3):
    result[i]=sorted([x for x in result[i] if x[0] not in d], key=lambda x:(x[1],x[0]), reverse=True)
    oo=open('result'+str(i)+'.txt','w')
    for w,p,s in result[i]:
        oo.write(w+'\t'+str(p)+'\t'+str(s)+'\n')
    oo.close()
