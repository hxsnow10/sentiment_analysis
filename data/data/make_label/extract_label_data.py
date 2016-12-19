# encoding=utf-8
import random
import multiprocessing as mp
import time
import itertools
from simple_senti import *
dsenti=Dict_rule_senti(senti_dict_path='senti_dict.txt', rule_path='rules.cfg')

def extract(inp, objects=[]):
    '''extract sents from text
       
    '''
    q,sents=inp
    rval1=[]
    rval2=[]
    for sent_ in sents:
        sent=sent_.strip()
        score,count= dsenti.senti_analysis(sent) 
        if score<=0.25 or score>=0.75:
            
            if score<0.25:
                s='-1'
            else:
                s='1'
            rval1.append(s+'\t'+sent+'\t'+str(count)+'\n')
        # elif 0.4<score<0.6 and len(rval1)>len(rval2):
        #        rval2.append('0'+'\t'+sent+'\t'+str(count)+'\n')
    rval=rval1
    q.put(rval)
    return len(rval)

def listener(q):
    '''listens for messages on the q, writes to file. '''

    f = open('label.data/label_sents.data', 'w') 
    while 1:
        m = q.get()
        if m == 'kill':
            f.write('killed')
            break
        for i in m:
            f.write(i)
        f.flush()
    f.close()

if __name__ == "__main__":
    file_path='unlabel_pred.data/pre_data2'
    #file_path='sss'
    
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(20)
    
    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    sents=open(file_path,'r')
        
    n=100000
    def sentss():
        while True:
            ss=[sents.readline() for i in range(n)]
            if ss:
                yield (q,ss)
            else:
                break
    t=sentss()
    N=60 
    while True:
        rr=pool.map(extract,itertools.islice(t,N))
        try:
            rr=pool.map(extract,itertools.islice(t,N))
        except:
            break
        print rr
        if rr==[]:break
    now=time.time()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()


