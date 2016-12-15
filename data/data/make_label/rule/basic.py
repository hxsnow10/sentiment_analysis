# encoding=utf-8
import numpy

fuhao=['：','，','。','！','？','#','!','?',',',';','；','—',':','…','.']

class Rule(object):
    '''Rule
    这里的Rule有几个组成部分，1）是否存在 需要匹配的模式 2）定位action的范围，执行action
    #1 匹配模式
    模式可能会冲突，比如 不，不一样，冲突的时候基本选最长匹配
    模式有时候会很复杂，比如存在歧义词，歧义的问题先放放（需要词性，DP,vector来消歧)；还有AA***bb这样的模式。
    现在的逻辑是：找出一个句子中所有的匹配，如果有覆盖，就选择最大的作为算子。
    另外这些算子加权有可能需要语法解析的结果

    match_key 
    find_updated_ran 需要被子类继承
    '''
    def __init__(self,ratio,postagger, dpparser, pos=False, dp=False):
       
        self.pos=pos
        self.dp=dp
        self.ratio=ratio
        self.postagger=postagger
        self.dpparser=dpparser

    def match_key(self, words, i, keys, postags=[], dparcs=[]):
        s=''
        k=i
        p=None
        while len(s)<=12 and k<len(words):# 最长匹配
            s+=words[k]
            if s in keys:
                print 'rule matched', s
                p=[i,k+1]
            k=k+1
        return p
 
    def find_updated_ran(self, words, key_ran, postags, dparcs):
        matched=False
        ran=[-1,-1]
        return matched,ran

    def pre(self, words):
        postags=None
        dparcs=None
        if self.pos:
            postags=self.postagger.transform(words)
        if self.dp:
            dparcs=self.dpparser.transform(words, postags)
        return postags, dparcs
    
    def match(self, words):
        postags,dparcs=self.pre(words)
        rval=[]
        for i in range(len(words)):
            print self, i
            key_ran=self.match_key(words, i, self.keys, postags, dparcs)
            if key_ran!=None:
                updated_rans=self.find_updated_ran(words,key_ran,postags, dparcs)
                for updated_ran,ratio in updated_rans:
                    rval.append((key_ran, updated_ran, ratio))
        #print rval
        return rval

class KeyRule(Rule):
    def __init__(self, keys, **kwargs):
        self.keys=keys
        super(KeyRule,self).__init__(**kwargs)

class NextDpWord(KeyRule):
    def __init__(self, next=[], **kwargs):
        keys=kwargs.pop('keys')        
        super(NextDpRule,self).__init__(keys, **kwargs)
        self.dp=True
        self.next=next

    def find_updated_ran(self, words, key_ran, postags=None, dparcs=None):
        for j,(head,relation) in enumerate(dparcs):
            if head==i and (relation in self.next or self.next==[]):
                return [[[j,j+1],self.ratio],[[i,i+1],0]]
        return [[[i,i+1],0]] 

class NextPosWord(KeyRule):
    def __init__(self, next=[], **kwargs):
        keys=kwargs.pop('keys')
        super(NextPosWord,self).__init__(keys,**kwargs)
        self.pos=True
        self.next=next
                
    def find_updated_ran(self, words, key_ran, postags, dparcs):
        k=key_ran[1]
        for j in range(k,len(words)):
            if words[j] in fuhao:break 
            if postags[j] in self.next or self.next==[]:
                return [[[j,j+1],self.ratio],[key_ran,0]]
        return [[key_ran,0]]

class NextSent(KeyRule):
        
    def find_updated_ran(self, words, key_ran, postags, dparcs):
        if key_ran[1]==len(words):return [[key_ran,0]]
        l=key_ran[1]
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

        return [[[l,r],self.ratio],[key_ran,0]]
    
class BeforeSent(KeyRule):
        
    def find_updated_ran(self, words, key_ran, postags, dparcs):
        if key_ran[0]==0:return [[key_ran,0]]
        l=key_ran[0]
        if l>0:
            while words[l] in fuhao:
                l-=1
                if l<0:
                    l=0
                    break
        r=l
        if r>0:
            while words[r] not in fuhao:
                r-=1
                if r<0:
                    break
        if r<0:r=0
        return [[[r,l],self.ratio],[key_ran,0]]

class AntiQuotesRule(Rule):

    def __init__(self):
        self.ratio=0
        self.pos=False
        self.dp=False
        self.keys=[]
    
    def match_key(self, words, i, keys, postags=[], dparcs=[]):
        next={'"':'"','“':'”'}
        if words[i] not in next:return None
        found=False
        for k in range(i+1,len(words)):
            if words[k]==next[words[i]]:
                found=True
                break
        if found and k-i<=4:
            return [i+1,k]
        else:return None
        
    def find_updated_ran(self, words, key_ran, postags, dparcs):
        return [[key_ran,self.ratio]]

class SequenceRule(KeyRule):
    def search_(self,key,words,i):
        assert words[i]==key[0]
        ran=[i]
        k=1
        j=i+1
        while k<len(key):
            find_k=False
            while j<len(words):
                if words[j]==key[k]:
                    find_k=True
                    break
                j+=1
            if not find_k:return None
            else:
                ran.append(j)
                k=k+1
        #print 'Seqence Rule',ran
        return ran
        
    def match_key(self, words, i, keys, postags=[], dparcs=[]):
        ran=None
        for key in keys:
            if words[i] !=key[0]:
                continue
            ran=self.search_(key,words,i)
            if ran!=None:return None
        return None 


class SequenceBetweenRule(SequenceRule):
    def find_updated_ran(self, words, key_ran, postags, dparcs):
        mmin,mmax=min(key_ran),max(key_ran)
        updated_ran=[]
        for i in range(mmin,mmax):
            if i not in key_ran:
                updated_ran.append(i)
        return [updated_ran,self.ratio]
        
#class 
