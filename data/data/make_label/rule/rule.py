#encoding=utf-8
import numpy
from basic import *
from nlp import *

def issubset(x,y):
    if type(x)!=type(set([])):
        xx=set(range(x[0],x[1]))
    else:xx=x
    if type(y)!=type(set([])):
        yy=set(range(y[0],y[1]))
    else:yy=y
    return xx<yy

class MulRule():
   
    def __init__(self, rule_path=None, ltp_model_path=None):
        self.postagger=postagger
        self.dpparser=dpparser
        self.rules=self.read_rules(rule_path) if rule_path!=None else []

    def transform(self,words,sen_len=10000):
        
        words=words[0:min(len(words),sen_len)]
        mul=numpy.ones([min(len(words),sen_len)])
        key_rans=[] 
        for rule in self.rules:
            print rule
            rans=rule.match(words)#ran=[key_ran, updated_ran ,ratio]
            key_rans=key_rans+rans
        print 'key_rans',key_rans
        label=[1,]*len(key_rans)
        for i in range(len(label)):
            for j in range(len(label)):
                if issubset(key_rans[i][0],key_rans[j][0]):
                    label[i]=0
        print 'label',label
        for i in range(len(key_rans)):
            if label[i]==1:
                key_ran, updated_ran ,ratio=key_rans[i] 
                mul[updated_ran[0]:updated_ran[1]]*=ratio
        return mul

    def read_rules(self,rule_path):
        fp=open(rule_path,'r')
        rules=[]
        while True:
            rule={}
            while True:
                s=fp.readline().strip()
                if s=='':
                    break
                s=s.split('=')
                rule[s[0].strip()]=eval(s[1].strip())
            if rule=={}:
                break
            type=rule.pop('type')
            rule['postagger']=self.postagger
            rule['dpparser']=self.dpparser
            if type=='nextword':
                rule=NextPosWord(**rule)
            elif type=='nextsent':
                rule=NextSent(**rule)
            elif type=='antiquote':
                rule=AntiQuotesRule()
            elif type=='sequencebetweenrule':
                rule=SequenceBetweenRule(**rule)
            elif type=='beforesent':
                rule=BeforeSent(**rule)
            else:rule=None
            if rule!=None:
                rules.append(rule)
        print rules
        return rules

class IndexRule():
   
    def __init__(self, vocab,sen_len=25, rule_path=None, ltp_model_path=None):
        self.postagger=postagger
        self.dpparser=dpparser
        self.vocab=vocab
        self.mulrule=MulRule(rule_path=rule_path, ltp_model_path=ltp_model_path)
        self.sen_len=sen_len

    def transform(self,indexs):
        words=[self.vocab[i] for i in indexs]
        mul=self.mulrule.transform(words,sen_len=self.sen_len)
        return mul
