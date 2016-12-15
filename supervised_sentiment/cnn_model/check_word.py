# encoding=utf-8
from cnn_model import *
from predict import *
from multiprocessing import Pool

m=allModel(vocab_path='vocab.txt',params_path='params.npz', )
ii=open('vocab.txt','r')
vocab=[]
for i in ii:
    vocab.append(i.strip().split()[0])

values=map(m.predict, vocab)
t=zip(vocab,values)
t=sorted(t, key=lambda x:x[1])
oo1=open('sorted_word_sentis1','w')
oo2=open('sorted_word_sentis2','w')

for w,p in t:
    if w in vocab:
        oo1.write(w+'\t'+str(p)+'\n')
    else:
        oo2.write(w+'\t'+str(p)+'\n')
        

