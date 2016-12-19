#encoding=utf-8
from textclf.model.pre import Preprocess

pre=Preprocess()
ii=open('data_','r')
oo=open('un.pre.data_','w')
for i in ii:
    a=i.strip().split('\t')
    w=''.join(a[1:])
    oo.write(pre.transform(w)[0]+'\n')
oo.close()

