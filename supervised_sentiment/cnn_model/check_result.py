# encoding=utf-8
import time
from predict import *
from numpy import mean
model=allModel('vocab2.txt',params_path='params.npz')
ii=open('data_b3','r')
N=0
senti_={'-1':[],'1':[]}
start=time.time()
oo=open('result.txt','w')
for i in ii:
    label,sent=i.split('\t')
    senti=model.predict(sent)
    if (senti-0.5)*float(label)>0:senti_[label].append(senti)
    if senti<0.5 and senti>0.45 and label=='-1':
        oo.write(sent+'\n')
    N=N+1
    if N%1000==0:
        now=time.time()
        oo.write(str(N)+'samples uses'+str(now-start)+'\n')
        oo.write('average time'+str((now-start)/N)+'\n')
        oo.write('-1 average'+str(sum(senti_['-1'])/len(senti_['-1']))+'\n')
        oo.write('-1 min'+str(min(senti_['-1']))+'\n')
        oo.write('-1 max'+str(max(senti_['-1']))+'\n')
        senti_['-1']=sorted(senti_['-1'],reverse=True)
        oo.write('-1 mean'+str(numpy.mean(senti_['-1'][:100]))+'\n')
        oo.write('1 average'+str(sum(senti_['1'])/len(senti_['1']))+'\n')
        oo.write('1 min'+str(min(senti_['1']))+'\n')
        oo.write('1 max'+str(max(senti_['1']))+'\n')
        oo.flush()
