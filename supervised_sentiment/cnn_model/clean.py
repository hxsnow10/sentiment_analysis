# encoding=utf-8
from random import shuffle

ii=open('data','r').readlines()
shuffle(ii)

oo=open('data2','w')
for i in ii:
    a=i.split('\t')
    if len(a)!=2:
        continue
    if a[0] not in ['1','-1']:
        print a[0]
        continue
    oo.write(i)
oo.close()


