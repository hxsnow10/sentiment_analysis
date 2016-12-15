# encoding=utf-8

ii=open('sorted_word.txt','r')
a=ii.readlines()
b=[]
cut=0.22

dd=open('senti_dict.txt', 'r')
d={}
for i in dd:
    if i[0]=='#':continue
    if i.strip()=='':continue
    w,p=i.strip().split()
    d[w]=p

oo=open('new_word.txt', 'w')
for i in a:
    f, w = i.strip().split()
    if float(f)<0.22 or float(f)>0.75:
        if float(f)<0.22:senti='-1'
        else:senti='1'
        if w not in d:
            oo.write(w+'\t'+str(senti)+'\n')
        elif senti!=d[w]:
            oo.write(w+'\t'+str(senti)+'\n')
