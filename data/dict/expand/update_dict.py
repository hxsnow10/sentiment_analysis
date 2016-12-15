# encoding=utf-8
import re

ii=open('senti_dict.txt', 'r')
s={}
for i in ii:
    w,p=i.strip().split()
    s[w]=p
d={}
cut=[0.8,1,0.8]
for i in range(3):
    label=str(i-1)
    if label=='0':
        print 'xxxxxxxxxxxxx'
        continue
    path='result'+str(i)+'.txt'
    ii=open(path,'r')
    for ss in ii:
        w,f=ss.strip().split()
        if f<=cut[i]:break
        if not re.match('\w+',w):
            if w not in s:d[w]=label
            if label=='0':print 'error'

d=d.iteritems()
d=sorted(d,key=lambda x:(x[1]))

oo=open('senti_dict2.txt', 'w')
for i in d:
    oo.write('\t'.join(i)+'\n')
oo.close()
