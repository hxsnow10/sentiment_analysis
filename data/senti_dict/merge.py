# encoding=utf-8

old=open('senti_dict.txt', 'r')
d={}
for i in old:
    if i[0]=='#':continue
    if i.strip()=='':continue
    w,p=i.strip().split()
    d[w]=p

new=open('in_result_true.txt','r')
for i in new.readlines():
    if i[0]=='#':continue
    if i.strip()=='':continue
    #j=i.decode('gbk').encode('utf-8')
    j=i
    try:
        w,p=j.strip().split('\t')
    except:
        print 'error'
    if w=='' or '//' in w or '@' in w:
        print w
        continue
    d[w]=p

oo=open('senti_dict2.txt','w')
for w in d:
    oo.write(w+'\t'+d[w]+'\n')


