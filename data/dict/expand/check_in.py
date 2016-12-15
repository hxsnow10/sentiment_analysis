# encoding=utf-8
import ahocorasick

# read seed senti dict
ii=open('senti_dict.txt','r')
d={}
for i in ii:
    w,p=i.strip().split()
    if p not in ['0','1','-1']:p='0'
    d[w]=p
ii.close()

A = ahocorasick.Automaton()
B = ahocorasick.Automaton()

for key in d:
                #print key,type(key)
    if len(key)>=len('好'):
        A.add_word(key, key)
        #print key
    if len(key)>len('好'):
        B.add_word(key, key)
    
A.make_automaton()
B.make_automaton()
#A.iter(text)
new_d={}
oo1=open('in_result_true.txt','w')
oo2=open('in_result_conflicts.txt','w')
for i in  range(3):
    if i=='1':continue
    name='result'+str(i)+'.txt'
    ii=open(name,'r')
    oo=open('result'+str(i)+'_noin.txt','w')
    for line in ii:
        w=line.split('\t')[0]
        if w in d:continue
        
        s=list(A.iter(w))
        sentis=[d[ww] for e,ww in s if d[ww]!='0']
        s_b=list(B.iter(w))
        sentis_b=[d[ww] for e,ww in s_b if d[ww]!='0']
        #if len(s)!=0 and len(sentis)!=0:
        if len(s)!=0 and len(sentis)!=0 and not (len(s_b)!=0 and len(sentis_b)!=0):
            ok=True
            for senti in sentis:
                if senti!=str(i-1):
                    ok=False
            if ok:
                oo1.write(w+'\t'+str(i-1)+'\n')
            else:
                oo2.write(w+'\t'+str(i-1)+'\t'+','.join(sentis)+'\n')
                oo.write(line)
        else:
            oo.write(line)

