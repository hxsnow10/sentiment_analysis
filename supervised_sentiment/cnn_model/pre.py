#encoding=utf8
import re
import jieba
import os
import sys
import datetime
from functools import partial
import jieba.posseg as pseg
reload(sys)
sys.setdefaultencoding('utf-8')

url = re.compile(u'http://[\w\/]+')
ref = re.compile(u'@\S{2,30}')

def pre_weibo(text):
    text=text.decode('utf-8')
    # remove strip simbols
    black_simbols=[u'　',u' ', u'\n']
    while len(text)>=1 and text[0] in black_simbols:
        text=text[1:]
    while len(text)>=1 and text[-1] in black_simbols:
        text=text[:-1]
    # 
    text = text.replace('&gt;','')

    # remove url
    text = re.sub(u'https?://[\w]+', '', text )
    #text = re.sub(u
    
    # text = re.sub(pattern9, '~', text )
    
    # remove @
    text = re.sub(ref, '', text )
    
    text=re.sub(u' \d+\.?\d* ', ' ', text)
    text=re.sub(u'[A-Za-z]+-?[0-9]+','',text)# 处理TB-230这样的产品型号
    text=re.sub(u'([0-9]{2,4}年)?([0-9]{1,2}月)?([0-9]{1,2}日)?([0-9]{1,2}[:：][0-9]{1,2})?','',text) #处理日期+数字
    '''
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'（','')
    text=text.replace(u'）','')
    text=text.replace(u'【','')
    text=text.replace(u'】','')
    '''
    return text.encode('utf-8')

def default_preprocess(x,lang='cn',stop_words=[], words=None, pos=False):
    con=lambda x:x not in stop_words
    if pos==False:
        cut=lambda x:jieba.cut(x)
        sent=' '.join(filter(con,[w.encode('utf-8') for w in cut(x)]))            
    elif pos==True:
        rval=list(pseg.cut(x))
        sent,poss=[],[]
        for i in range(len(rval)):
            w,p=rval[i]
            w=w.encode('utf-8')
            p=p.encode('utf-8')
            if w not in stop_words:
                sent.append(w)
                poss.append(p)
        sent=' '.join(sent)
        poss=' '.join(poss)
    if not pos:
        return sent
    else:
        return sent,poss

class Preprocess:
    def __init__(self,type='default',user_pre=None,userdict_path=None,stop_words_path=None,words=None,params={}):
        '''
        type 'user':user_pre callable,params
        type 'default':userdict_path,stop_words_path,words
        type 'None':do Nonthing
        '''
        self.dict=userdict_path
        if self.dict!=None:
            jieba.load_userdict(self.dict)
        
        self.stop_words=['',' ','  ','\n','\r','\t']
        if stop_words_path!=None:
            self.stop_words=self.stop_words+[i.strip() for i in open('stop_words_path','r')]
        self.stop_words=set(self.stop_words)
        self.params=params
        if words!=None:
            self.words=set(words)
        else:
            self.words=None
        
        if type=='default':
            self.pre=partial(default_preprocess,stop_words=self.stop_words,words=self.words)
        elif type=='default_pos':
            self.pre=partial(default_preprocess,stop_words=self.stop_words,words=self.words, pos=True)
        elif type=='None':
            self.pre=lambda x:x.strip()

    def transform(self,X, pos=False):
        if type(X)==type([1,2]):
            X_new=[self.pre(i) for i in X]
            #TODO: 由于一开始分词肯定是没有vocabulary的，所以不能使用vocabulary。
        elif type(X)==type('xxx'):
            X_new=[self.pre(X)]#ext input must a list
        return X_new

    def fit(self,X,y=None):
        return self

    def get_params(self,deep=True):
        return {}
        
