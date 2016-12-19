# encoding=utf-8


ii=open('pre_data','r')
oo=open('pre_data2','w')

for i in ii:
    try:
        j=i.decode('utf-8')
        oo.write(i)
    except:
        pass

