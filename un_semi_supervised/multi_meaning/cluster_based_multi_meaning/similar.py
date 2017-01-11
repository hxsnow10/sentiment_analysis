# encoding=utf-8
from gensim.models import word2vec
import numpy
from copy import deepcopy
from numpy import ndarray
import time
from multiprocessing import Pool

model=word2vec.Word2Vec.load_word2vec_format('multi_vector.txt',binary=False)
while True:
    try:
        w=raw_input('Enter word:').decode('utf-8')
        if w not in model:
            print 'word not in model'
            continue
        n=raw_input('Enter n:(default 20)')
        if n.strip()=='':n=20
        else:n=int(n.strip())
        similar=model.most_similar(w, topn=n)
        for ww,p in similar:
            print ww,p
    except Exception,e:
        print 'error',e
        break
'''
# give result
result=[[],[],[]]
conflict=[]
for w in new_d:
    if (new_d[w]!=numpy.array([0,0,0])).any():
        score=new_d[w]/(sum(new_d[w])+0.00001)
        index=ndarray.argmax(score)
        probs=score[index]
        if w in d: 
            if str(index-1)!=d[w]:
                conflict.append((w.encode('utf-8'), d[w], str(index-1), str(probs)))
                
        else: result[index].append((w,float(probs)))
oo=open('conflict.txt','w')
conflict=sorted(conflict, key=lambda x:float(x[-1]),reverse=True)
for i in conflict:
    oo.write('\t'.join(i)+'\n')
oo.close()

print [len(i) for i in result]
for i in range(3):
    result[i]=sorted(result[i], key=lambda x:(x[1],x[0].encode('utf-8')), reverse=True)
    oo=open('result'+str(i)+'.txt','w')
    for w,p in result[i]:
        oo.write(w.encode('utf-8')+'\t'+str(p)+'\n')
    oo.close()
'''
