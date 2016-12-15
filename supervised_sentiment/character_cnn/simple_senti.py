# encoding=utf-8
'''
simple sentiment analyisi based dict and rules
'''
from pre import Preprocess
from rule import Rule
#dict based
from functools import partial
class Dict_rule_senti():

    def __init__(self, senti_dict_path, rule_path=None):
        self.pre=Preprocess()
        ii=open(senti_dict_path,'r')
        a=ii.readlines()
        self.senti_dict={}
        for i in a:
            w,p=i.strip().split()
            self.senti_dict[w]=p
        if rule_path!=None:
            self.rule=Rule(rule_path=rule_path)

    def senti_analysis(self, sent, pre=True):
        if pre:
            sent=self.pre.transform(sent.strip())[0]
        words=sent.strip().split(' ')
        pp,nn=[],[]
        def senti(w, d):

            if w not in d:return 0
            if d[w]=='-1':
                print w,'-1'
                return -1
            if d[w]=='1':
                print w,'1'
                return 1
            return 0
        sentis=map(partial(senti, d=self.senti_dict),words)
        count=reduce(lambda x,y:x+abs(y),sentis,0)
        mul=self.rule.transform(words)
        r=[sentis[i]*mul[i] for i in range(len(sentis)) if sentis[i]!=0]
        r=[i for i in r if i!=0]
        score=0.5*sum(r)/(len(r)+0.001)+0.5
        # score=0.5*merge(sentis)+0.5
        return score,count

def merge(a):
    a=[float(i) for i in a]
    if a==[]:return 0.5
    ans1=min(a)
    def weight(x):
        if abs(x-0.5)<0.1:return 0
        else:return (x-0.5)*(x-0.5)
    w=map(weight,a) 
    if sum(w)==0:ans2=0.5
    else:
        ans2=sum([i*j for i,j in zip(a,w)])*1.0/(sum(w))
    return ans2

if __name__=='__main__':
    d=Dict_rule_senti(senti_dict_path='senti_dict.txt', rule_path='rules.cfg')
    while True:
        sent=raw_input('Text=')
        print d.senti_analysis(sent)
