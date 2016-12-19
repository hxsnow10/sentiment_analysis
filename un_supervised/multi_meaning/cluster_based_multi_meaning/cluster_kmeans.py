# encoding=utf-8
'''
基于k_means,考虑能否加速
'''

from gensim.models import word2vec
import numpy as np
from sklearn.cluster import k_means
#config
window=5
half=window/2
k=3
vocab_path='vocab.txt'
data_path='un.data'
vec_path='vec2.txt'
new_data_path='new_data'
import time

# CPU
# 这里有几个操作：get_data(numpy +), predict(norm,-), train()

def get_vocab(vocab_path):#O(0)
    ii=open(vocab_path,'r')
    vocab=[]
    for i in ii:
        vocab.append(i.strip().split()[0])
    return vocab

def vectorize(line,word2index,vector):#O(n),n是line 长度
    words=line.strip().split()
    words=[w for w in words if w in word2index]
    words=['</s>',]*half+words+['</s>',]*half
    indexs=[word2index[w] for w in words]
    return words,vector[indexs]

def get_data(data, vocab, vector):#Lines*(window*n*100+read_time),能并行的地方就是把里面的计算并行
    word2index={w:i for i,w in enumerate(vocab)}
    new_data={w:[] for w in vocab}
    for line in data:
        line,vecs=vectorize(line,word2index,vector)
        print len(line),vecs.shape
        wcs=[(line[i],sum(vecs[range(i-half,i)+range(i+1,i+half+1)])) for i in range(half,len(line)-half)]
        for w,contexts in wcs:
            new_data[w].append(contexts)
        
    return new_data

def get_new_word(w,context,centroids):#k*100
    if centroids==[]:return w
    start=time.time()
    #print w,context
    dist=np.array([np.linalg.norm(centroids[i]-context) for i in range(k)])
    kk=np.argmax(dist)
    #print time.time()-start
    return w+'_'+str(kk)


def label(data, vocab, vector, centroid):#这里可以并行
    word2index={w:i for i,w in enumerate(vocab)}
    new_data=[]
    N=0
    for line in data:
        N=N+1
        line,vecs=vectorize(line,word2index,vector)
        new_line=' '.join([get_new_word(line[i],sum(vecs[range(i-half,i)+range(i+1,i+half+1)]),centroid[line[i]]) for i in range(half,len(line)-half)])
        new_data.append(new_line)
    return new_data

def main():
    start=time.time()
    vocab=get_vocab(vocab_path)
    model=word2vec.Word2Vec.load_word2vec_format(vec_path,binary=False)
    vector=np.array([model[w.decode('utf-8')] for w in vocab if w.decode('utf-8') in model])
    data=open(data_path,'r').readlines()
    all_data=get_data(data,vocab, vector)#TODO
    print len(all_data)
    #print all_data['轮子']
    print time.time()-start
    centroid={w:[] for w in vocab}
    oo=open('centroids.txt','w')
    for w in all_data:#TODO这里似乎不能CPU并行，考虑GPU
        w_data=all_data[w]
        if len(w_data)<=2:
            print w,'0 samples'
            oo.write(w+'\t'+str([])+'\n')   
            continue
        cen,_,_=k_means(w_data, n_clusters=k, init='k-means++', precompute_distances='auto', n_init=10, max_iter=100, verbose=False, tol=0.0001, random_state=None, copy_x=True, n_jobs=20, return_n_iter=False)
        centroid[w]=cen
        oo.write(w+'\t'+str(cen)+'\n')   
    #print centroid
    print time.time()-start
    data=label(data,vocab,vector,centroid)#TODO
    oo=open(new_data_path,'w')
    print time.time()-start
    for i in data:
        oo.write(i+'\n')

if __name__=='__main__':
    main()
