# encoding=utf-8
"""Convolutional network example.

Run the training for 50 epochs with
```
python __init__.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
import logging
import numpy
import Queue
from argparse import ArgumentParser

from fuel.datasets.hdf5 import H5PYDataset
import theano
import numpy as np
from theano import tensor
from blocks.bricks.base import application
from blocks.algorithms import GradientDescent, Scale, AdaDelta, Adam
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation, BatchNormalization)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.lookup import LookupTable
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate,SquaredError
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph, apply_dropout, apply_batch_normalization, get_batch_normalization_updates
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.utils import shared_floatx_nans
from blocks.initialization import IsotropicGaussian
from blocks.filter import VariableFilter
from blocks.roles import add_role,INPUT,OUTPUT,DropoutRole,WEIGHT

from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from toolz.itertoolz import interleave

from textclf.model.theano import IndexMul
from textclf.vocab.words import Vocabulary
import theano.tensor as T

class Skip_Gram_Multi(Initializable):
    
    def __init__():
        

    def apply(self, con ,sen):
        '''
        batch inputs of sents, transform to context_con_vector, sense_cen_vectors, 最后隐射到附近是context的概率。
        '''
        def cos_sim(v1, v2):
            numerator = theano.tensor.sum(v1*v2)
            denominator = theano.tensor.sqrt(theano.tensor.sum(v1**2)*theano.tensor.sum(v2**2))
            return numerator/denominator
        con_vectors=self.context_lookup.apply(con)
        cen_vectors=self.center_lookup.apply(cen)
        sen_vectors=self.mmeans_lookup.apply(cen)
        probs=cos_sim(con_vectors.sum(axis=1),cen_vectors)

        costs=1/(-T.exp(sen_vectors*con_vectors))# sen_v: b*vec_len  con_v:b*(2*window+1)*con_len  ==> b*(2*window+1)
        # 2 type of final costs:1)consider probs 2)consider argmax of probs
        final_costs=probs*costs
        return probs,final_costs
        #TODO
        
class Lookup_Ruled(Initializable):
    '''
        
    '''

    def __init__(self, words_num=10000, vector_len=100, word_vectors_path=None, 
            rule_path=None, vocab_path=None, sen_len=25, 
            # weights_init=None,
            **kwargs):
        super(Lookup_Ruled, self).__init__(use_bias=False)
        self.words_num=words_num
        self.vector_len=vector_len
        self.word_vectors_path=word_vectors_path
        self.vocab_path=vocab_path
        if self.vocab_path!=None:
            self.get_vocab(vocab_path)
            self.words_num=len(self.vocab)
        self.rule_path=rule_path
        self.sen_len=sen_len
        self.enable_rule=False
        if rule_path!=None:
            self.rule=IndexMul(vocab=self.vocab ,sen_len=sen_len, rule_path=rule_path)
            self.enable_rule=True
        self.weights_init=IsotropicGaussian()
        self.has_bias=False
    
    def get_vocab(self, vocab_path):
        ii=open(vocab_path,'r')
        vocab=[]
        for i in ii:
            vocab.append(i.strip().split()[0])
        self.vocab=vocab
        return vocab

    def _allocate(self):
        self.W = shared_floatx_nans((self.words_num, self.vector_len),
                name='WordVectoe')
        add_role(self.W, WEIGHT)
        self.parameters.append(self.W)
        # self.add_auxiliary_variable(self.W.norm(2), name='WordVector_norm')
    
    def _initialize(self):
        W,=self.parameters
        if self.word_vectors_path!=None:
            W.set_value(self.init_word_vectors)
        else:
            self.weights_init.initialize(W, self.rng)
    
    @application(input=['inputs'],output=['rval'])
    def apply(self, inputs):
        rval=self.W[inputs]
        rval=T.cast(rval.dimshuffle(0,'x',1,2),'float32')
        return rval
words_num=0
class Huffman_Tree(Initializable):

    def __init__(self, vocab_path, vector_len=100):
        super(Huffman_Tree, self).__init__(use_bias=False)
        freq=self.get_freq(vocab_path)
        words_num=len(freq)
        self.vector_len=vector_len
        self.build_tree(freq)
        '''
        self.words2code={}
        self.id2theta={}
        self.left={}
        self.right={}
        '''
    def get_freq(self, vocab_path):
        ii=open(vocab_path,'r')
        freq=[]
        for i in ii:
            w,c=i.strip().split()
            c=int(c)
            freq.append((w,c))
        return freq

            
    def build_tree(self, freq):
        p = Queue.PriorityQueue()
        leaves=[HuffmanNode(name=l[0],value=l[1],leaf=True) for l in freq]
        for l in leaves:    # 1. Create a leaf node for each symbol
            p.put(l)             #    and add it to the priority queue
        while p.qsize() > 1:         # 2. While there is more than one node
            l, r = p.get(), p.get()  # 2a. remove two highest nodes
            node = HuffmanNode(left=l, right=r,value=l.value+r.value) # 2b. create internal node with children
            p.put(node) # 2c. add new node to queue      
        self.root=p.get()
        self.leaves=leaves               # 3. tree is complete - return root node
        self.nolf_nodes=self.root.all_nodes(lambda x:not x.leaf)
        for k in range(len(self.nolf_nodes)):
            self.nolf_nodes[k].id=k
    
    def _allocate(self):
        n=len(self.nolf_nodes)
        self.W = shared_floatx_nans((n, self.vector_len,2),
                name='Hs_vector')
        add_role(self.W, WEIGHT)
        self.parameters.append(self.W)
     
    def _initialize(self):
        W,=self.parameters
        self.weights_init.initialize(W, self.rng)
    
    @application(input=['h'],output=['y_hat'])
    def apply(self, h):
        y_hat=self.build_probs(h)
        return y_hat

    def build_probs(self,inp):
        probs=theano.shared(np.array([1,]*100))
        todo=[[self.root,probs]]
        start,end=0,1
        while start<end:
            node,probs=todo[start]
            node.root_probs=probs
            if not node.leaf:
                new_probs=probs.dimshuffle(0,'x')*T.nnet.softmax(T.dot(inp, self.W[node.id,:,:]))
                todo.append([node.left,new_probs[:,0]])
                end+=1
                todo.append([node.right,new_probs[:,1]])
                end+=1
            start+=1
        probs=[node.root_probs for node in self.leaves]
        probs=T.stack(probs).dimshuffle(1,0)
        return probs

class HuffmanNode(object):
    def __init__(self, leaf=False, name=None, value=None, left=None, right=None, childs=[]):
        self.leaf=leaf
        self.name=name
        self.value=value
        self.left = left
        self.right = right
        self.childs = []
        self.id=None
        self.probs_root = None# prob of root to this node
    
    def all_nodes(self, filter_func):
        '''中序'''
        l=[self]
        start,end=0,1
        while start<end:
            node=l[start]
            if node.left!=None and filter_func(node):
                l.append(node.left)
                end+=1
            if node.right!=None and filter_func(node):
                l.append(node.right)
                end+=1
            start+=1
        rval=filter(filter_func, l)
        return rval
        

def main(save_to='cnn.pkl', num_epochs=500, batch_size=100,
        data_path='data.hdf5',
        words_num=817, vector_len=100, word_vectors_path=None,
        rule_path=None, vocab_path='vocab.txt', sen_len=25,# config LookUp_Ruled
        filter_sizes=[(1,100),(2,100),(3,100),(4,100),(5,100)], num_filters=[100,100,50,50,25],# config multi_conv
        top_mlp_activations=[Softmax()],top_mlp_dims=[325,2],# config top mlp
        
        ):


    _train=H5PYDataset(data_path,which_sets=('train',))
    _test=H5PYDataset(data_path,which_sets=('test',))
    
    train_stream = DataStream.default_stream(
        _train, iteration_scheme=ShuffledScheme(
            _train.num_examples, batch_size))
    test_stream = DataStream.default_stream(
        _test,  iteration_scheme=ShuffledScheme(
            _test.num_examples, batch_size))
    num_batches=_train.num_examples/batch_size
    
    con = tensor.lmatrix('con')
    cen = tensor.lmatrix('cen')
    s=theano.shared(np.zeros((words_num),dtype=np.float32))
    def one_hot(k):
        return T.set_subtensor(s[k],1.0)
    ncen,_=theano.map(fn=one_hot,sequences=cen)
    lookup=Lookup_Ruled(words_num=words_num, vector_len=100, word_vectors_path=None,
            rule_path=None, vocab_path=None, sen_len=25,)
    huffman_tree=Huffman_Tree(vocab_path=vocab_path)
    lookup.initialize()
    huffman_tree.initialize()
    hid=lookup.apply(con).sum(axis=2).dimshuffle(0,2)
    probs=huffman_tree.apply(hid)
    # Normalize input and apply the convnet


    cost = (SquaredError().apply(ncen, probs)
            .copy(name='cost'))
    #return locals()
    error_rate = (MisclassificationRate().apply(ncen.argmax(axis=1), probs)
                  .copy(name='error_rate'))
    cg = ComputationGraph([cost, error_rate])
    Ws = VariableFilter(roles=[WEIGHT])(cg.variables)
    for W in Ws:
        cost = cost + .00005 * (W ** 2).sum()
    cost.name='final_cost'
    # apply dropuout
    #inputs=VariableFilter(theano_name='softmax_apply_input_')(cg.variables)
    #add_role(inputs[0],DropoutRole)
    #cg_dropout = apply_dropout(cg, inputs, 0.5)

    # apply batch normilization
    #cg = apply_batch_normalization(cg)
    # Add updates for population parameters
    # pop_updates = get_batch_normalization_updates(cg)
    #extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
    #     for p, m in pop_updates]
    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Adam(learning_rate=0.02))
    # algorithm.add_updates(extra_updates)
    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      test_stream,
                      prefix="test"),
                  ProgressBar(),
                  Printing(),
               TrainingDataMonitoring(
                  [cost, error_rate,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True)]

    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)
    main_loop.run()

