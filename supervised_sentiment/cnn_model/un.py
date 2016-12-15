# encoding=utf-8

ii=open('data','r')
oo=open('un.data','w')
for i in ii:
    a=i.split('\t')
    oo.write(''.join(a[1:]))
