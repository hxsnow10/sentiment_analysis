# encoding=utf-8


ii=open('senti_dict.txt','r')
vocab=[]
for i in ii:
    vocab.append(i.strip().split('\t')[0])
voacb=set(vocab)

oo2=open('sorted_word_sentis2','w')
for i in open('sorted_word_sentis1','r'):
    w,p=i.strip().split()
    if w not in vocab and (float(p)<0.45 or float(p)>0.55):
        if float(p)<0.45:p=-1
        else:p=1
        oo2.write(w+'\t'+str(p)+'\n')
