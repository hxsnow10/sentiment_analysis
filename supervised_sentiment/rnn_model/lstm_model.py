"""Convolutional network example.

Run the training for 50 epochs with

It is going to reach around 0.8% error rate on the test set.

"""
import logging
import numpy
from argparse import ArgumentParser
from blocks import initialization

from fuel.datasets.hdf5 import H5PYDataset
import theano
import numpy as np
from theano import tensor
from blocks.bricks.base import application
from blocks.algorithms import GradientDescent, Scale, AdaDelta, Adam, CompositeRule, StepClipping,RMSProp
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,Linear,
                           Softmax, Activation, BatchNormalization)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                            Flattener, MaxPooling)
from blocks.bricks.lookup import LookupTable
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks.recurrent.misc import Bidirectional
from blocks.bricks.recurrent.architectures import GatedRecurrent , LSTM
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph, apply_dropout, apply_batch_normalization, get_batch_normalization_updates
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.utils import shared_floatx_nans,shared_floatx_zeros
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
from functools import partial
from theano import tensor

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
    
    @application(input=['state_blow'],output=['rval'])
    def apply(self, state_blow):
        add_role(state_blow,DropoutRole)
        state_blow=T.cast(state_blow,'int64')
        inputs=state_blow[:,1:]
        lengths=state_blow[:,0]
        s=theano.shared(np.zeros((self.sen_len),dtype=np.float32))
        print 'get in fprop' 
        def fn(x):
            rval=T.set_subtensor(s[:x],1.0)
            return rval 
        def fn2(x):
            return self.rule(x)
     
        mask,_=theano.map(fn=fn,sequences=lengths)
        if self.enable_rule:
            multiply,_=theano.map(fn=fn2,sequences=inputs)
            print 'syntax enabled'
            mask=mask*multiply
        rval=self.W[inputs]
        mask=T.shape_padright(mask,rval.ndim-2)
        rval=rval*mask
        rval=T.cast(rval.dimshuffle(1,0,2),'float32')
        return rval
    

class SentiNet(Initializable):
    """LeNet-like convolutional network.
    """
    def __init__(self, words_num, vector_len, words_vectors_path, 
            rule_path, vocab_path, sen_len, 
            filter_sizes, num_filters,
            top_mlp_activations, top_mlp_dims, batch_size, **kwargs):
        super(SentiNet, self).__init__(**kwargs)
        
        self.lookup=Lookup_Ruled(words_num=words_num, vector_len=vector_len, 
                vocab_path=vocab_path, rule_path=rule_path,sen_len=sen_len)
        self.linear=Linear(input_dim=vector_len, output_dim=vector_len * 4, name='linear_bofore_lstm')
        self.bdir= Bidirectional(
                    prototype=LSTM(dim=100))
        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.bn1=BatchNormalization(input_dim=(batch_size,vector_len))
        self.bn2=BatchNormalization(input_dim=(top_mlp_dims[0]))

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims, weights_init=IsotropicGaussian(),
                biases_init=Constant(1))
        self.children=[self.lookup, self.bn1, self.bdir, self.bn2, self.top_mlp, self.linear]

    @application(input='inputs',output='rval')
    def apply(self, inputs):
        inputs=self.lookup.apply(inputs)
        inputs=self.linear.apply(inputs)
        inputs,_=self.bdir.apply(inputs)
        return self.top_mlp.apply(tensor.max(inputs,0))
        #return self.top_mlp.apply(inputs[0]+inputs[-1])

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

def main(save_to='rnn.pkl', num_epochs=50, batch_size=50,
        data_path='data.hdf5',
        words_num=200000, vector_len=100, word_vectors_path=None,
        rule_path=None, vocab_path='vocab.txt', sen_len=25,# config LookUp_Ruled
        filter_sizes=[(1,100),(2,100),(3,100),(4,100),(5,100)], num_filters=[100,100,50,50,25],# config multi_conv
        top_mlp_activations=[Softmax()],top_mlp_dims=[200,2],# config top mlp
        return_model=False 
        ):
    
    sn=SentiNet(batch_size=batch_size, words_num=words_num, vector_len=vector_len, words_vectors_path=word_vectors_path, 
            rule_path=rule_path, vocab_path=vocab_path, sen_len=sen_len, 
            filter_sizes=filter_sizes, num_filters=num_filters,
            top_mlp_activations=top_mlp_activations, top_mlp_dims=top_mlp_dims
            )
    print sn.weights_init
    print sn.biases_init
    initialize([sn])
    print sn.weights_init
    print sn.biases_init
    sn.initialize()
    x = tensor.lmatrix('X') 
    y = tensor.lmatrix('y')
    # Normalize input and apply the convnet
    probs = sn.apply(x)
    model = Model(probs)
    if return_model:
        return model
    
    cost = (CategoricalCrossEntropy().apply(y, probs)
            .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.argmax(axis=1), probs)
                  .copy(name='error_rate'))
    cg = ComputationGraph([cost, error_rate])
    '''
    Ws = VariableFilter(roles=[WEIGHT])(cg.variables)
    for W in Ws:
        cost = cost + .00005 * (W ** 2).sum()
    cost.name='final_cost'
    '''
    # apply dropuout
    # inputs=VariableFilter(theano_name='softmax_apply_input_')(cg.variables)
    # add_role(inputs[0],DropoutRole)
    # cg_dropout = apply_dropout(cg, inputs, 0.5)

    # apply batch normilization
    # cg = apply_batch_normalization(cg)
    # Add updates for population parameters
    # pop_updates = get_batch_normalization_updates(cg)
    #extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
    #     for p, m in pop_updates]
    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([AdaDelta(),StepClipping(1.0)]))
    '''
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([Scale(0.01)])
    )
    '''
    '''
    step_rules = [RMSProp(learning_rate=0.002, decay_rate=0.95),
              StepClipping(1.0)]
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=CompositeRule(step_rules))
    '''
    _train=H5PYDataset(data_path,which_sets=('train',), subset=slice(0,3000))
    _test=H5PYDataset(data_path,which_sets=('test',), subset=slice(0,100))
    
    train_stream = DataStream.default_stream(
        _train, iteration_scheme=ShuffledScheme(
            _train.num_examples, batch_size))
    test_stream = DataStream.default_stream(
        _test,  iteration_scheme=ShuffledScheme(
            _test.num_examples, batch_size))
    checkpoint = Checkpoint(save_to, after_training=False)
    # checkpoint.add_condition(['after_epoch'],
    #                                 predicate=OnLogRecord('test_final_cost'))
    gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
    step_norm = aggregation.mean(algorithm.total_step_norm)
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate, aggregation.mean(algorithm.total_gradient_norm), step_norm],
                      train_stream,
                      prefix="train"),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      test_stream,
                      prefix="test"),
                  checkpoint,
                  ProgressBar(),
                  Printing(),
               TrainingDataMonitoring(
                  [cost, error_rate,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True)]

    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)
    main_loop.run()

