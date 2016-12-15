#encoding=utf8
import jieba
from functools import partial
from pyltp import Segmentor as _Segmentor
from pyltp import Postagger as _Postagger
from pyltp import Parser as _Parser

class Segmentor():

    def __init__(self, model_path):
        self.segmentor = _Segmentor()  # 初始化实例
        self.segmentor.load(model_path+'cws.model')  # 加载模型

    def transform(self, sent):
        words = self.segmentor.segment(sent)  # 分词
        stop=[' ','','\t','\n','  ']
        return [x for x in list(words) if x not in stop]

class Postagger():

    def __init__(self, model_path):
        postagger = _Postagger() # 初始化实例
        print 'dasdad',model_path
        postagger.load(model_path+'pos.model')  # 加载模型
        self.postagger=postagger

    def transform(self, words):
        return self.postagger.postag(words)
        
class Dpparser():

    def __init__(self, model_path):
        dpparser = _Parser() # 初始化实例
        dpparser.load(model_path+'parser.model')  # 加载模型
        self.dpparser=dpparser

    def transform(self, words, postaggs=None):
        return self.dpparser.parse(words, postaggs)

segmentor=Segmentor('ltp/ltp_data/')
postagger=Postagger('ltp/ltp_data/')
dpparser=Dpparser('ltp/ltp_data/')
# thrift start
# postagger=Postagger('opinion/ltp/ltp_data/')
# dpparser=Dpparser('opinion/ltp/ltp_data/')
if __name__=='__main__':
    while True:
        text=raw_input('Enter Text:')
        
        words=segmentor.transform(text)
        postags=postagger.transform(words)
        arcs=dpparser.transform(words, postags)
        names=['ROOT']+words
        postags=['NONE']+list(postags)
        arcs=[(None,None)]+[(arc.head, arc.relation) for arc in arcs]
        print 'names=','\t'.join([str(k)+':'+i for k,i in enumerate(names) if k>=1])
        print 'postags=','\t'.join([str(k)+':'+i for k,i in enumerate(postags) if k>=1])
        print 'arcs=','\t'.join([' '.join((str(k),names[k],names[i[0]],str(i[1]))) for k,i in enumerate(arcs) if k>=1])
        
