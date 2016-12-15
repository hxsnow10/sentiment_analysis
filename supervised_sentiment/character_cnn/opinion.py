# encoding=utf8
import random
from random import uniform
import json
import yaml
from predict import allModel
from rel_extract import rel_extract, ltp_dependency_grammar
from functools import partial
from simple_senti import *
import json

class OpinionMiner():

    def __init__(self,vocab_path,params_path,ltp_model_path,senti_dict_path=None,rules_path=None, nn_path=None):
        self.load_objects()
        self.senti_model=allModel(vocab_path=vocab_path,params_path=params_path,
                senti_dict_path=senti_dict_path, rules_path=rules_path, nn_path=nn_path, pos_path=ltp_model_path+'pos.model')
        self.get_senti=self.senti_model.predict
        dp=ltp_dependency_grammar(path=ltp_model_path)
        self.rel_extract=partial(rel_extract,dp=dp.transform)

    def load_objects(self):
        self.objects=None
        self._obj2obj=None
        return None,None

    def analysis(self, title, content, objects=None, objects_updated=False):
        if objects==None:
            if objects_updated:
                objects,_obj2obj=load_objects()
            else:
                objects=self.objects
                _obj2obj=self._obj2obj
        # extract key sentence from text
        else:
            _obj2obj={}
            for obj in objects:
                for _obj in objects[obj]:
                    _obj2obj[_obj]=obj
        all_objects=[]
        for obj in objects:
            all_objects+=objects[obj]
        text=title+content
        sents, ranges, objectss=self.extract_sents(text, all_objects)
        print sents,ranges,objectss
        # for every sent, get [(obj, senti),...]
        sent_opinion=[self.get_opinion(sent, objs) for sent,objs in zip(sents,objectss)]
        # do merge, get {obj:(final_senti, [(range, sneti),...]), ...}
        rval={}
        for i in range(len(ranges)):
            ran_=ranges[i][0]
            for s in sent_opinion[i]:
                _obj,senti,ran=s
                obj=_obj2obj[_obj]
                if obj not in rval:
                    rval[obj]=[0.5,[]]
                rval[obj][1].append(((ran[0]+ran_,ran[1]+ran_),senti))
        for obj in rval:
            senti_final=self.merge([ele[1] for ele in rval[obj][1]])
            rval[obj][0]=senti_final
        
        rval['default']=self.get_senti(title)
        rval=json.dumps(rval, ensure_ascii=False)
        return rval
    
    def merge(self, a):
        a=[float(i) for i in a]
        # print a
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
    
    def extract_sents(self, text, objects):
        '''extract key sents from text about objects
        TODO: performance . loop about word/n-gram ranther than 
        '''
        key_words=objects
        sents,ranges,objectss, start,i=[],[],[],0,0
        text=text.decode('utf-8')
        l={0:0}
        for i in range(1,len(text)+1):
            l[i]=l[i-1]+len(text[i-1].encode('utf-8'))
        n=len(text)
        cut_simbols=[u'.',u'。',u'!',u'！',u'?',u'？',u'#',u'】',u' ',u'　']
        i=0
        while i<n:
            if text[i] in cut_simbols or i==n-1:
                while i<n and text[i] in cut_simbols or i==n-1:i+=1
                oo=[]
                for key in key_words:
                    if key.decode('utf-8') in text[start:i]:
                        oo.append(key)
                if oo!=[]:
                        sents.append(text[start:i].encode('utf-8'))
                        ranges.append((l[start],l[i]))
                        objectss.append(oo)
                start=i
                i=i-1
            i+=1
        return sents,ranges,objectss
    
    def get_opinion(self, sent, objects):
        rval=[]
        relations=self.rel_extract(sent, objects)
        sentis=[self.get_senti(rel['phrase'], pre=False) for rel in relations]# TODO:get_senti 并行最好放外面
        for i in range(len(sentis)):
            obj= relations[i]['obj']
            ram= relations[i]['ram']
            anti=relations[i]['anti']
            senti=sentis[i]
            if anti:
                senti=-senti
            rval.append((obj,senti,ram))
        return rval
    
    def simple_rel_extract(self, sent, objects):#TODO
        rval=[]
        for obj in objects:
            rval.append({'obj':obj,'phrase':sent})
        return rval
    
    def simple_get_senti(self, phrase):#TODO
        return uniform(0,1)

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

if __name__=='__main__':
    a=OpinionMiner(
        vocab_path='vocab.txt',
        params_path='params.npz',
        ltp_model_path='ltp/ltp_data/') 
    text='京东好。淘宝好。马云傻逼。'
    objects={'电商':['京东','淘宝'],'马云':['马云']}
    objects_updated=False
    print a.analysis(text, text, objects, objects_updated)
    while True:
        title=raw_input('Title:')
        content=raw_input('Content:')
        objects=raw_input('Object:')
        # names=raw_input('
        print a.analysis(title, content, {objects:[objects]})

