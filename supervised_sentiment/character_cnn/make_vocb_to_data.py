# encoding=utf-8

ii=open('senti_dict.txt','r')
oo=open('test','w')
for i in ii:
    w,p=i.strip().split()
    oo.write(p+'\t'+w+'\n')
    
