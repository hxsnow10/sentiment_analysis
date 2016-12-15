# encoding=utf-8
'''
count pos/neg of label_data
'''

ii=open('label_sents.data','r')
oo=open('un.data_','w')
N=0
pos,neg=0,0
while True:
    i=ii.readline()
    if i[0]=='#':continue
    try:
        a=i.strip().split()
        label,score,c=a[0],a[1],a[2]
        sent='\t'.join(a[3:])
        if label=='-1':
            neg+=1
        else:
            pos+=1
        oo.write(sent+'\n')
    except:
        pass
        # print i.strip().split() 
    N=N+1
    if N%1000000==0:
        print N
        print pos,neg
