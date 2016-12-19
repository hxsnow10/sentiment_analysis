# encoding=utf-8
import re
import cPickle as pickle

def parse_hanyucidian_meaning(line):
    '''一般的有2种方法，正则与自动机，自动机可以处理上下文相关的解析，如果正则规则强一点也能处理
    '''
    line=line.decode('utf-8')
    s=re.search(u'【([\u4e00-\u9fa5]+)】', line)
    if not s:return None, None
    word=s.groups()[0]
    s=len(word)+1+1
    p=s
    while p< len(line) and not re.match(u'[①②③\u4e00-\u9fa5～，。｜？！；]',line[p]):
        p+=1
    pronounce=line[s:p]
    meanings=line[p:]
    meanings=meanings.replace(u'①','\t').replace(u'②','\t').replace(u'③','\t')
    meaningss=meanings.split('\t')
    ee=[]
    for m in meaningss:
        try:
            explanation,examples=m.split(u'：')
            examples=examples.split('｜')
            ee.append([explanation, examples])
        except:
            pass
    info={'pronounce': pronounce, 'meanings': ee}
    print word.encode('utf-8'),meanings.encode('utf-8')
    return word, info

if __name__=='__main__':
    ii=open('现代汉语词典（第五版）全文.txt','r')
    dict_info={}
    for line in ii:
        if line.strip()!='':
            word,info=parse_hanyucidian_meaning(line)
            if not word:continue
            #print word
            dict_info[word]=info
    pickle.dump(dict_info, open('现代汉语词典.pkl','w'))
    #dict_info=
