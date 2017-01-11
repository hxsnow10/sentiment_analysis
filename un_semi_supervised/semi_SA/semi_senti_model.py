"""Convolutional network example.

Run the training for 50 epochs with
```
python __init__.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
import logging
import numpy
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
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.predicates import OnLogRecord
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

class Multi_Prototype_Lookup_Ruled(Initializable):
    '''
        
    '''

    def __init__(self, words_num=10000, vector_len=100, word_vectors_path=None, prototype_num=1,
            rule_path=None, vocab_path=None, sen_len=25, context_size=3,  
            # weights_init=None,
            **kwargs):
        super(Lookup_Ruled, self).__init__(use_bias=False)
        self.words_num=words_num
        self.vector_len=vector_len
        self.word_vectors_path=word_vectors_path
        if self.word_vectors_path!=None:
            params=numpy.load(word_vectors_path)
            self.word_vectors=params['vectors']
            self.word_probs=params['probs']
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
        self.W = shared_floatx_nans((self.words_num, self.prototype_num, self.vector_len),
                name='WordVector')
        add_role(W, WEIGHT)
        self.probs = shared_floatx_nans((self.words_num, self.prototype_num), name='ProtoProbs')
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
        def fn(x):
            rval=T.set_subtensor(s[:x],1.0)
            return rval 
        def fn2(x):
            return self.rule(x)
        def get_vector():
     
        mask,_=theano.map(fn=fn,sequences=lengths)
        if self.enable_rule:
            multiply,_=theano.map(fn=fn2,sequences=inputs)
            print 'syntax enabled'
            mask=mask*multiply
        # rval=self.W[inputs] # here to change 
        contexts=[]
        reshape(inputs)
        rval=theano.map(get_vector,sequences=[self.W[inputs],self.probs[inputs],contexts])
        mask=T.shape_padright(mask,rval.ndim-2)
        rval=rval*mask
        rval=T.cast(rval.dimshuffle(0,'x',1,2),'float32')
        return rval
    

class MultiConv(Initializable):
    '''
    '''
    def __init__(self, filter_sizes, num_filters, pool_sizes, batch_size=100, **kwargs):
        assert len(filter_sizes)==len(pool_sizes)
        super(MultiConv, self).__init__(**kwargs)
        # calcutate pool_sizes
        self.children=\
            [ConvolutionalSequence(
                layers=[Convolutional(filter_size=filter_sizes[i],
                            num_filters=num_filters[i],
                            name='conv_{}'.format(i),
                            batch_size=batch_size,
                            ),
                        
                        MaxPooling(pooling_size=pool_sizes[i],name='pool_{}'.format(i)
                        )],
                batch_size=batch_size,
                num_channels=1,
                )
             for i in range(len(filter_sizes))]
        print self.children
    
    @application(input=['x'],output=['output'])
    def apply(self, x):
        add_role(x,DropoutRole)
        for child in self.children:
            ys=[child.apply(x) for child in self.children]
        return T.concatenate(ys, 1)

class SentiNet(FeedforwardSequence, Initializable):
    """LeNet-like convolutional network.
    """
    def __init__(self, words_num, vector_len, words_vectors_path, 
            rule_path, vocab_path, sen_len, 
            filter_sizes, num_filters,
            top_mlp_activations, top_mlp_dims, batch_size, **kwargs):
        
        self.lookup=Lookup_Ruled(words_num=words_num, vector_len=vector_len, 
                vocab_path=vocab_path, rule_path=rule_path,sen_len=sen_len)
        pool_sizes=[(sen_len-i+1,vector_len-j+1) for i,j in filter_sizes]
        self.conv= MultiConv(filter_sizes=filter_sizes, pool_sizes=pool_sizes, num_filters=num_filters, batch_size=batch_size)
        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        self.bn1=BatchNormalization(input_dim=(1,sen_len,vector_len))
        self.bn2=BatchNormalization(input_dim=(top_mlp_dims[0]))

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims, weights_init=IsotropicGaussian(),
                biases_init=Constant(1))

        application_methods = [self.lookup.apply, self.bn1.apply, self.conv.apply, self.flattener.apply,self.bn2.apply,
                               self.top_mlp.apply]
        super(SentiNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

def main(save_to='cnn.pkl', num_epochs=10, batch_size=100,
        data_path='data.hdf5',
        words_num=200000, vector_len=100, word_vectors_path=None,
        rule_path=None, vocab_path='vocab.txt', sen_len=25,# config LookUp_Ruled
        filter_sizes=[(1,100),(2,100),(3,100),(4,100),(5,100)], num_filters=[100,100,50,50,25],# config multi_conv
        top_mlp_activations=[Softmax()],top_mlp_dims=[325,2],# config top mlp
        return_model=False    
        ):
    
    sn=SentiNet(batch_size=batch_size, words_num=words_num, vector_len=vector_len, words_vectors_path=word_vectors_path, 
            rule_path=rule_path, vocab_path=vocab_path, sen_len=sen_len, 
            filter_sizes=filter_sizes, num_filters=num_filters,
            top_mlp_activations=top_mlp_activations, top_mlp_dims=top_mlp_dims
            )
    un_model=#TODO
    X_label = tensor.lmatrix('X_label')
    y_label = tensor.lmatrix('y_label')
    X_unlabel = tensor.lmatrix('X_unlabel')
    sn.initialize()
    # Normalize input and apply the convnet
    probs = sn.apply(x)
    model = Model(probs)
    if return_model:
        return model

    supervised_cost = (CategoricalCrossEntropy().apply(y, probs)
            .copy(name='cost'))
    unsupervised_cost = un_model.cost(X_label) 
    cost = supervised_cost + unsupervised_cost
    error_rate = (MisclassificationRate().apply(y.argmax(axis=1), probs)
                  .copy(name='error_rate'))
    cg = ComputationGraph([cost, error_rate])
    Ws = VariableFilter(roles=[WEIGHT])(cg.variables)
    for W in Ws:
        cost = cost + .00005 * (W ** 2).sum()
    cost.name='final_cost'
    _train=H5PYDataset(data_path,which_sets=('train',))
    _test=H5PYDataset(data_path,which_sets=('test',))
    
    train_stream = DataStream.default_stream(
        _train, iteration_scheme=ShuffledScheme(
            _train.num_examples, batch_size))
    test_stream = DataStream.default_stream(
        _test,  iteration_scheme=ShuffledScheme(
            _test.num_examples, batch_size))
    
    # apply dropuout
    # inputs=VariableFilter(theano_name='softmax_apply_input_')(cg.variables)
    # add_role(inputs[0],DropoutRole)
    # cg_dropout = apply_dropout(cg, inputs, 0.5)

    # apply batch normilization
    cg = apply_batch_normalization(cg)
    # Add updates for population parameters
    # pop_updates = get_batch_normalization_updates(cg)
    #extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
    #     for p, m in pop_updates]
    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=AdaDelta())
    # algorithm.add_updates(extra_updates)
    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    
    checkpoint = Checkpoint(save_to, after_training=False)
    checkpoint.add_condition(['after_epoch'],
                                     predicate=OnLogRecord('test_final_cost'))
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      test_stream,
                      prefix="test"),
                  checkpoint,
                  ProgressBar(),
                  Printing()]
    '''
               TrainingDataMonitoring(
                  [cost, error_rate,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True),
    '''
    
    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)
    main_loop.run()

