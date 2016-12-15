# encoding=utf-8

ii=open('data_b3','r')
oo=open('data_c','w')
for i in ii:
    try:
        w,s=i.strip().split('\t')
        oo.write(w+'\t'+s.replace(' ','')+'\n')
    except Exception,e:
        print e
oo.close()
