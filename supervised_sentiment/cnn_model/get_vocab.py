# encoding=utf-8

from collections import OrderedDict
cut=15

ii=open('un.data','r')
oo=open('vocab.txt','w')
vocab=OrderedDict()
for sent in ii:
    words=sent.strip().split()
    for word in words:
        if word not in vocab:
            vocab[word]=0
        else:
            vocab[word]+=1
for w in vocab:
    if vocab[w]>=cut:
        oo.write(w+'\n')
    
