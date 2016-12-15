# encoding=utf-8

ii=open('senti_dict.txt', 'r')
d={}
for i in ii:
    w,p=i.strip().split()
    d[w]=p
d=d.iteritems()
d=sorted(d,key=lambda x:(x[1],x[0]))

oo=open('senti_dict2.txt', 'w')
for i in d:
    oo.write('\t'.join(i)+'\n')
oo.close()


