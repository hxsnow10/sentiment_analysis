# encoding=utf-8
import json
import requests
from BeautifulSoup import BeautifulSoup

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
        # get meanings
        try:
            rval['meaning']=[]
            a=data.find('div',{'class':"content means imeans",'id':"detailmean-wrapper",'data-group':"imeans",'style':"display: none"}).findAll('li')
            for x in a:
                mm=[i.text.encode('utf-8') for i in x.findAll('p')]
                rval['meaning'].append(mm)
        except Exception,e:
            print 'find meaning',e
        # get 近义词/反义词
        try:#TODO：有时候只有近义词或者反义词，需要识别出来
            s=[x.text.encode('utf-8') for x in data.find('div',{'class':"content module",'id':"syn_ant_wrapper"}).findAll('label')]
            a=data.find('div',{'class':"content module",'id':"syn_ant_wrapper"}).findAll('div', {'class':"block"})
            for i in range(len(s)):
                if s[i]=='近义词':
                    rval['list0']=[x.text.encode('utf-8') for x in a[i].findAll('a')]
                elif s[i]=='反义词':
                    rval['list1']=[x.text.encode('utf-8') for x in a[i].findAll('a')]
            print json.dumps(rval,ensure_ascii=False)
        except Exception,e:
            print 'find words_list',e
    except:
        print 'not found or error'
        print data
    # get 百度解释
    return rval

def catch_meaning_from_baidu(word):
    params={'wd':'表示','device':'pc','from':'home'}
    s=requests.get('http://dict.baidu.com/s',params=params)
    print s,len(s.text)
    if len(s.text)==11217:
        print s.text
    data=s.text
    info=baidu_meaning_parser(data)
    return info

def get_vocab(path='vocab.txt'):
    vocab=[]
    for line in open(path,'r'):
        vocab.append(line.strip().split()[0])
    return vocab

def main():
    vocabs=get_vocab()
    rval={}
    for w in vocabs:
        print w
        rval[w]=catch_meaning_from_baidu(w)

if __name__=='__main__':
    main()
