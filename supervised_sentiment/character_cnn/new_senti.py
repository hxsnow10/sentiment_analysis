# encoding=utf-8


ii=open('vocab.txt','r')
d={}
for i in ii:
    w=i.strip().split()[0]
    d[w]='1'

ii=open('senti_dict2.txt','r')

oo=open('new_senti_dict.txt','w')
dd={}
for i in ii:
    w,p=i.strip().split()
    if w not in d:
        oo.write(w+'\t'+p+'\n')
    dd[w]=p

