# encoding=utf-8
from multiprocessing import Pool,Lock, Value
import multiprocessing as mp
from functools import partial
import random
import time
import itertools
from nlp import Segmentor
from pre import pre_weibo2 as pre_weibo

seg=Segmentor('ltp/ltp_data/')

def extract_sents(inputs):
    '''extract sents from text
       
    '''
    #print '1'
    q,sents=inputs
    rval=[]
    for sent_ in sents:
        sent=sent_.strip()
        sent=sent.replace('\r\n','。')
        sent=sent.replace('\n','。')
        sent=pre_weibo(sent)
        pre_data=' '.join(seg.transform(sent))+'\n'
        rval=rval+[pre_data]

    q.put(rval)
    return len(rval)

def listener(q):
    '''listens for messages on the q, writes to file. '''

    f = open('unlabel_pred.data/pre_data_nocut', 'w') 
    while 1:
        m = q.get()
        if m == 'kill':
            f.write('killed')
            break
        for i in m:
            f.write(i)
        f.flush()
    f.close()

def mapto(a):
    l=map(len,a)
    b=reduce(lambda x,t:x+t,a,[])
    return b,l

if __name__=='__main__':
    todo=['out_buzz.data','out_news.data','sohu.data']
    #todo=['out_buzz.test.data','out_news.test.data','sohu.test.data']
    
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(21)
    
    #put listener to work first
    watcher = pool.apply_async(listener, (q,))
    
    for file_path in todo:
        start=time.time()
        texts=open('unlabel.data/'+file_path,'r')
        n=10000
        def textss():
            while True:
                ss=[]
                for i in range(n):
                    s=texts.readline()
                    if s!='':ss.append(s)
                if ss:
                    yield (q,ss)
                else:
                    break
        t=textss()
        N=60
        while True:
            try:
                sentss=pool.map(extract_sents,itertools.islice(t,N))
            except Exception,e:
                print 'error',e
                break
            print sentss
            if sentss==[]:break
        now=time.time()
        print now-start
