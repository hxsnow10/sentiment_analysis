from pre import Preprocess
from rule import Rule
from functools import partial
import ahocorasick

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
        self.A=self.buildA(keys)

    def buildA(self, keys):
        A = ahocorasick.Automaton()
        for key in objects:
            print key,type(key)
            A.add_word(key, key)
        A.make_automaton()
        return A 
    
    def transform(self, sent):
        locss=[[key,end_idx-len(key)+1,end_idx+1] for end_idx, key in self.A.iter(text)]
         
        return locss
