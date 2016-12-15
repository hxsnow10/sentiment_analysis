# encoding=utf-8

i1=open('data','r')
i2=open('un.pre.data_','r')
oo=open('data','w')

while True:
    a=i1.readline()
    b=i2.readline()
    s=a.split()[0]+'\t'+b
    oo.write(s)
