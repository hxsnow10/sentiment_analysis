# encoding=utf-8
import jieba.posseg as pseg 

ii=open('senti_dict2.txt','r')
oo=open('senti_dict_n.txt','w')
d={}
for i in ii:
    w,p=i.strip().split()
    d[w]=p
for w in d:
    words = list(pseg.cut(w))
    if len(words)!=1:continue
    word,flag=words[0]
    if flag=='n' and d[w]=='-1':
        word=word.encode('utf-8')
        oo.write(word+'\t'+d[word]+'\n')


