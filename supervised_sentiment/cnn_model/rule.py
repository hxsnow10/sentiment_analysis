#encoding=utf-8
import numpy
from jieba import posseg

class Rule():
   
    def __init__(self, sen_len=50, rule_path=None,flag=None):
        '''

        Params
        ------------
        rule_path: path of rule
        vocabulary: dict. str(word):int(index) 词典是模型的首要核心。
        flag: dict. str(word):str(flag) 需要人工对词性进行纠正。

        '''
        self.rules=self.read_rules(rule_path) if rule_path!=None else []
        keys=[rule['key'] for rule in self.rules]
        self.keys=set(reduce(lambda x,y:x+y, keys, []))
        self.sen_len=sen_len


    def transform(self,words,poss=None):
        mul=numpy.ones([len(words)])
        # if set(words) | self.keys== set([]):
        #     return mul
        for i in range(len(words)):
            w=words[i]
            for rule in self.rules:
                if rule['type']=='cw':#单名单,作用域:本身加权
                    if w in rule['key']:
                        mul[i]*=rule['ratio']
                        # print 'cw'
                ''' 
                if rule['type']=='cf':
                    if self.index2flag[w] in rule['key']:#匹配词性 作用域：本身加权
                        mul[i]*=rule['ratio']
                '''
                if rule['type']=='cw_nw':#匹配单词 作用域：本身变为0，下一个形容词加权乘
                    if w in rule['key']:
                        if i+1<len(words):mul[i+1]*rule['ratio']
                        '''
                        k=self.find_next_word(poss,i,rule['next'])
                        if k!=-1:
                            mul[k]*=rule['ratio']
                        '''
                        # print 'cw_nw'
                if rule['type']=='cw_ns':#匹配单词 作用域：本身变为0，下一短句加权乘
                    if w in rule['key']:
                        l,r=self.find_next_sen(words,i)
                        if l<len(words):
                            mul[l:r]*=rule['ratio']
                        # print 'cw_ns'
        return mul

    def read_rules(self,rule_path):
        print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
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
            rules.append(rule)
        return rules

    def find_next_word(self, words, index, tarList):
        for i in range(index + 1, len(words)):
            if i in tarList:
                return i
        return -1

    def find_next_sen(self, words, index):
        '''
        从第一个非标点符号开始，
        到下一个一个标点符号为止。标点符号为分句符号：逗号，句号。
        由于简陋，只能这样了。
        '''
        fuhao=['，','。','！','？','#','!','?']
        l=index+1
        if l<len(words):
            while words[l] in fuhao:
                l+=1
                if l>=len(words):
                    break
        r=l
        if r<len(words):
            while words[r] not in fuhao:
                r+=1
                if r>=len(words):
                    break

        return l,r


