# encoding=utf-8
import xlrd

def get_ts(path):
    data  = xlrd.open_workbook(path)
    table = data.sheets()[0]
    words=table.col_values(0)
    values=table.col_values(1)
    return zip(words, values)

d={}
for i in open('senti_dict.txt','r'):
    w,p=i.strip().split()
    d[w]=p
n=0
new=get_ts('new.xlsx')
for w,p in new:
    #print w,type(w),p,type(p)
    try:w=w.encode('utf-8')
    except:print 'error'
    p=str(int(p))
    if w not in d:print w
    elif d[w]!=p:
        n=n+1
        d[w]=p 

print n
oo=open('senti_dict2.txt','w')
for w in d:
    oo.write(w+'\t'+str(d[w])+'\n')
oo.close()   
