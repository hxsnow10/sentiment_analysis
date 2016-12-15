# encoding=utf-8
import requests
import readline
from pyltp import Segmentor, Postagger, Parser
from pre import Preprocess

class ltp_dependency_grammar():
    def __init__(self, path='./ltp/ltp_data/'):
        self.segmentor = Preprocess()  # 初始化实例
        self.postagger = Postagger() # 初始化实例
        self.postagger.load(path+'pos.model')  # 加载模型
        self.parser = Parser() # 初始化实例
        self.parser.load(path+'parser.model')  # 加载模型
        
    def transform(self, sent):
        words = self.segmentor.transform(sent)[0].split()  # 分词
        postags = self. postagger.postag(words)  # 词性标注
        arcs = self. parser.parse(words, postags)  # 句法分析
        words=['ROOT']+list(words)
        arcs=[(None,None)]+[(arc.head, arc.relation) for arc in arcs]
        return words, list(postags), arcs

def rel_extract(sent, objects, dp=ltp_dependency_grammar().transform):
    names,_, arcs=dp(sent)#
    print 'names=','\t'.join([str(i[0])+':'+i[1] for i in enumerate(names)])
    print 'arcs=','\t'.join([str(i) for i in arcs])
    n=len(names)
    nexts={}
    for id in range(1, n):
        head,relation=arcs[id]
        print names[id],head,relation
        if head not in nexts:
            nexts[head]=[]
        if id not in nexts:
            nexts[id]=[]
        if relation=='SBV':
            nexts[id].append(head)
            nexts[head].append(id)
        elif not (relation=='COO' and '、' not in names[min(id,head):max(id,head)]) and not (relation=='IS'):
            nexts[id].append(head)
            nexts[head].append(id)
            
    rval=[]
    print 'next=',nexts
    for obj in objects:
        linked=[]
        linked_set=set([])
        start,end=0,0
        for i in range(1, n):
            if (names[i] in obj) or (obj in names[i]):
                linked.append(i)
                end+=1
                linked_set.add(i)
        mi,ma=10000,1# two selections: a [min,ma] or selective words connect
        while(start<end):
            id=linked[start]
            print id
            for j in nexts[id]:
                if j not in linked_set:
                    print '\t',j
                    linked.append(j)
                    linked_set.add(j)
                    mi=min(mi,j)
                    ma=max(ma,j)
                    end+=1
            start+=1
        mi=max(mi,1)
        ss=''.join(names[mi:ma+1])
        l={1:0}
        ll={0:1}
        # print names
        for i in range(2,len(names)+1):
            l[i]=l[i-1]+len(names[i-1])
            ll[l[i]]=i
        mi=l[mi]
        ma=l[ma+1]
        # print mi,ma
        anti=False
        cmp1=['比']
        cmp2=['不比','没有','不如','不像']
        c=cmp2+cmp1
        o=ss.find(obj)
        for c_ in c:
            i=ss.find(c_)
            if i>=0:
                if i+len(c_)==o and c_ in cmp1:
                    anti=True
                elif o+len(obj)==i and c_ in cmp2:
                    anti=True
                    mi=i+len(c_)
                elif i+len(c_)==o and c_ in cmp2:
                    mi=i+len(c_)
                break 
        # print mi,ma
        s=' '.join(names[ll[mi]:ll[ma]])
        rval.append({'obj':obj, 'phrase':s, 'ram':(mi,ma+1), 'anti':anti})
    return rval
 
def search(i, names, nexts, obj, linked):
    if linked[i]!=-1:return linked
    linked[i]=1
    print names[i],linked[i],nexts[i]
    for j in nexts[i]:
        if linked[j]==-1:
            linked=search(j, names, nexts, obj, linked)
    return linked


if __name__=='__main__':
    ltp_dp=ltp_dependency_grammar()
    while True:
        text=raw_input('Enter text:')
        objects=[raw_input('Enter object:')]
        for i in rel_extract(text,objects,ltp_dp.transform):
            print 'object=\t',i['obj']
            print 'phrase=\t',i['phrase']
            print 'anti=\t',i['anti']

    
