# encoding=utf-8
import json
import requests
from BeautifulSoup import BeautifulSoup
import logging
import traceback
import multiprocessing as mp

logging.basicConfig(level=logging.DEBUG,
                format='%(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='baidu_dict_crawler.log',
                filemode='a+')


def baidu_meaning_parser(data):
    data=BeautifulSoup(data)
    rval={}
    '''
    rval['meaning']=[]
    a=data.find('div',{'class':"content means imeans",'id':"detailmean-wrapper",'data-group':"imeans",'style':"display: none"}).findAll('li')
    for x in a:
        mm=[i.text.encode('utf-8') for i in x.findAll('p')]
        rval['meaning'].append(mm)
    s=[x.text.encode('utf-8') for x in data.find('div',{'class':"content module",'id':"syn_ant_wrapper"}).findAll('label')]
    a=data.find('div',{'class':"content module",'id':"syn_ant_wrapper"}).findAll('div', {'class':"block"})
    for i in range(len(s)):
        if s[i]=='近义词':
            rval['list0']=[x.text.encode('utf-8') for x in a[i].findAll('a')]
        elif s[i]=='反义词':
            rval['list1']=[x.text.encode('utf-8') for x in a[i].findAll('a')]
    print json.dumps(rval,ensure_ascii=False)
    '''
    try:
        rval['meaning']=[]
        a=data.find('div',{'class':"content means imeans",'id':"detailmean-wrapper",'data-group':"imeans",'style':"display: none"}).findAll('li')
        for x in a:
            mm=[i.text.encode('utf-8') for i in x.findAll('p')]
            rval['meaning'].append(mm)
        try:
            s=[x.text.encode('utf-8') for x in data.find('div',{'class':"content module",'id':"syn_ant_wrapper"}).findAll('label')]
            a=data.find('div',{'class':"content module",'id':"syn_ant_wrapper"}).findAll('div', {'class':"block"})
            for i in range(len(s)):
                if s[i]=='近义词':
                    rval['list0']=[x.text.encode('utf-8') for x in a[i].findAll('a')]
                elif s[i]=='反义词':
                    rval['list1']=[x.text.encode('utf-8') for x in a[i].findAll('a')]
        except:
            pass
        return rval,None
    except Exception, e:
        logging.error('find words_list error:'+str(traceback.format_exc()))
        error=traceback.format_exc()
        print error
        return rval,error
    # get 百度解释

def catch_meaning_from_baidu(word):
    #word='的'
    params={'wd':word,'device':'pc','from':'home'}
    try:
        s=requests.get('http://dict.baidu.com/s',params=params)
    except:
        return '','error'
    print s,len(s.text)
    if len(s.text)==11217:
        return '','error'
        # print s.text.encode('utf-8')
        oo=open('error.html','w')
        oo.write(str(s.text.encode('utf-8')))
        oo.close()
    data=s.text
    info,error=baidu_meaning_parser(data)

    print word,json.dumps(info,ensure_ascii=False)
    return info,error

def get_vocab(path='vocab.txt'):
    vocab=[]
    for line in open(path,'r'):
        vocab.append(line.strip().split()[0])
    return vocab

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

def main():
    vocabs=get_vocab()
    oo=open('baidu_dict.txt','w')
    rval={}
    #VOCABS is shared by all process
    #every process get task from vocabs, process , write result/log, and update VOCABS
    manager = mp.Manager()
    q = manager.Queue()
    q2=manager.Queue()
    pool=mp.Pool(20)
    watcher = pool.apply_async(listener, (q,))

    def process(p):
        q,w=p
        info,error=catch_meaning_from_baidu(w)
        if error:
            # TODO if not in: pass
            vocabs.append(w)
            logging.error('wrong'+str(w))
            q.put(w)
        else:
            logging.info('ok'+str(w))
            oo.write(w+'\t'+str(json.dumps(info,ensure_ascii=False))+'\n')
    while q:
        sub_words=q.get()
        if len(sub_words)<100: break
        results=pool.map(process,[[q,w] for w in sub_words])
    
if __name__=='__main__':
    main()
