# encoding=utf-8
import random
import multiprocessing as mp
import time
import itertools
from simple_senti_debug import *
import json
import traceback
import pprint

dsenti=Dict_rule_senti(senti_dict_path='senti_dict.txt', rule_path='rules.cfg')

def extract(inp, objects=[]):
    '''extract sents from text
       
    '''
    q,sents=inp
    rval1=[]
    rval2=[]
    for sent_ in sents:
        sent=sent_.strip()
        score,info1, info2= dsenti.senti_analysis(sent)
        if info1 and info1['local_conflicts'] + info1['global_conflicts']:
            rval1.append('norule_debug_info:\t'+str(json.dumps(info1,ensure_ascii=False,indent=4))+'\n')
        if info2 and info2['local_conflicts'] + info2['global_conflicts']:
            rval2.append('rule_debug_info:\t'+str(json.dumps(info2,ensure_ascii=False,indent=4))+'\n')
    q.put((rval1,rval2))
    return len(rval1+rval2)

def worker(p):
    try:
        return extract(p)
    except:
        traceback.print_exc()
        raise

def listener(q):
    '''listens for messages on the q, writes to file. '''

    f1 = open('debug/debug1.txt', 'w') 
    f2 = open('debug/debug2.txt', 'w') 

    while 1:
        m=q.get()
        if m == 'kill':
            f.write('killed')
            break
        m1,m2=m
        for i in m1:
            f1.write(i)
        f1.flush()
        for i in m2:
            f2.write(i)
        f2.flush()

if __name__ == "__main__":
    file_path='unlabel_pred.data/pre_data'
    #file_path='sss'
    
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(20)
    
    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    sents=open(file_path,'r')
        
    n=100
    def sentss():
        while True:
            ss=[sents.readline() for i in range(n)]
            if ss:
                yield (q,ss)
            else:
                break
    t=sentss()
    N=20
    while True:
        rr=pool.map(worker,itertools.islice(t,N))
        print rr
        if rr==[]:break
    now=time.time()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()


