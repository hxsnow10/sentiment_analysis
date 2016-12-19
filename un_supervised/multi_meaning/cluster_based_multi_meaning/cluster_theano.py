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
import theano.tensor as T

class Kmeans(Initializable):
    '''
        
    '''
    def __init__(self, prototype_num=3, vec_path=None, sen_len=25, window_size=5,  
            # weights_init=None,
            **kwargs):
        super(Kmeans, self).__init__(use_bias=False)
        self.vocab,self.word_vectors=self.read_word_vector(vec_path)
        print 'shape of word_vector', self.word_vectors.shape
        self.words_num, self.vector_len=self.word_vectors.shape
        self.sen_len=sen_len
        self.window_size=window_size
        self.prototype_num=prototype_num
        self.weights_init=IsotropicGaussian()
        self.has_bias=False
    
    def read_word_vector(self, vec_path):
        ii=open(vec_path,'r')
        ii.readline()
        vocab=[]
        word_vectors=[]
        for i in ii:
            s=i.strip().split()
            w,vec=s[0],[float(x) for x in s[1:]]
            vocab.append(w)
            word_vectors.append(vec)
        word_vectors=np.array(word_vectors)
        return vocab,word_vectors

    def _allocate(self):
        self.W=theano.shared(self.word_vectors)
        self.Centroids= shared_floatx_nans((self.words_num, self.prototype_num, self.vector_len),
                name='WordCentroids')
        add_role(self.Centroids,WEIGHT)
        self.parameters.append(self.Centroids)
        # self.add_auxiliary_variable(self.W.norm(2), name='WordVector_norm')
    
    def _initialize(self):
        W,=self.parameters
        self.weights_init.initialize(W, self.rng)
    
    @application(input=['state_blow'],output=['rval'])
    def cost(self, state_blow):
        state_blow=T.cast(state_blow,'int64')
        inputs=state_blow[:,1:]
        lengths=state_blow[:,0]

        s=theano.shared(np.zeros((self.sen_len),dtype=np.float32))
        def fn(x):
            rval=T.set_subtensor(s[:x],1.0)
            return rval  
        mask,_=theano.map(fn=fn,sequences=lengths)

        # rval=self.W[inputs] # here to change 
        sent_vector=self.W[inputs]
        context=sent_vector
        half=self.window_size/2
        context=T.zeros_like(sent_vector)
        L=self.sen_len
        for k in range(-half,half):
            if k!=0:
                context=T.inc_subtensor(context[:,max(0,k),min(L,L+k)],sent_vector[:,max(0,-k),min(L,L-k)])
        for i in range(half):
            context=T.set_subtensor(context[:,i],context[:,i]/(i+half))
            context=T.set_subtensor(context[:,-i-1],context[:,i-1]/(i+half))
        context=T.set_subtensor(context[:,half+1:L-half],context[:,half+1:L-half]/(2*half))
        centroids=self.Centroids[inputs]
        norm=((centroids-context.dimshuffle(0,1,'x',2))**2).sum(3)
        # print norm.shape.eval()
        dist=norm.min(2)*mask
        label=T.argmin(norm,2)*mask
        cost=T.sum(dist)
        return cost,label
    
def main(save_to='cluster.pkl', num_epochs=200, batch_size=100,
        data_path='data.hdf5_test',
        vec_path='vec.txt', sen_len=25,window_size=5,prototype_num=3,# config LookUp_Ruled
        return_model=False
        ):
    kmeans=Kmeans(vec_path=vec_path, sen_len=sen_len, window_size=window_size, prototype_num=prototype_num)
    x = tensor.imatrix('X')
    kmeans.initialize()
    # Normalize input and apply the convnet
    cost,label = kmeans.cost(x)
    model = Model(label)
    cg = ComputationGraph(cost)
    if return_model:
        return model
    
    _train=H5PYDataset(data_path,which_sets=('train',))
    _test=H5PYDataset(data_path,which_sets=('develop',))
     
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
    
    checkpoint = Checkpoint(save_to, after_training=False, save_main_loop=False)#TODO: only save parameters
    checkpoint.add_condition(['after_epoch'],
                                     predicate=OnLogRecord('develop_kmeans_cost_output_0'))
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost],
                      test_stream,
                      prefix="develop"),
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

