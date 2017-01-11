'''
Classes to process Tree_data or graph_data especially in theano-like language.

####Why and What

Network/Graph is the most generative and common structure in __our Modeling 
World__.If we say there is a graph here, it means all should be represented by 
nodes and links, where 'all' equals this model_system info in every place and every time.

We don't say graph model equal reality, for example: RNN with specific parameters
could understand sentences, however traditional parsing tree(like CG, DG, UG, 
witout vector) can only be a partial view and can never understand sentences.
__It seems both symbolic and Tensor has their places.__
（并非说这样的计算流就是OK的了，比如在自然语言理解，一个词的语义需要结合上下文去确定，
这就体现人的神经网络的能力，或者说一个简单的RNN利用自身空间结构与时间上动态处理的能力
最终表现出的能力）

RNN is most popular graph with tensor.Modern(2016) Computing of RNN in computer
is some different from bololgy brain.That's a big question but not talked here.

Recursive NN is some computing based on paring tree.I don't think it's a complete
path for language understanding, in fact, right parsing need right right language 
understanding( which also applies for words segmentation and Pos Tagging).
But when i try to implement RecursiveNN, i ask if it is a common request when 
__vector computing constrainted in predefined explict graph/structure__.RNN is 
graph structure implicit.I think it is a common request, whether symbolic input partial
or output partial.许多场景，symbolic数据是自然存在的。

So it is a package about Structural Tensor Inference.

####CG,DG,UG
CG=Constiunent Grammar
DG=Denpency Grammar
UG=Universial Grammar

C tree 是把原来的序列解析为一颗组合树，叶子节点=原来序列，非叶子节点是儿子的组合性节点
    * 关于结构的计算方式是自然的，算子具体怎么定义 TODO
        * 一般来说还是2个儿子吧
        * Pos 参与
        * 
    * 拓展：虽然计算依赖实际上是跨树的，打破树结构并不合适（或者说应该RNN动态隐含地生成树），可以在生成imbedding的时候考虑依赖
D tree 是把原来的序列解析为一颗树，所有节点=原来序列，边描述了节点之间的关系
    * 计算方式，当一个节点有多个儿子的时候，怎么计算？

####Objectives
理论与实现上，我们有以下目标
1. 实现基于graph的显式vecotr infernece
2. 理论上显式与隐式Inference的区别
3. 关于生成graph
4. 我想知道RNN与symbolic的表达能力的边界

####Implementation:
也许这个问题需要深入到语言层与硬件层才能良好的解决，先尝试一下。

先不过度优化
'''

'''
比如我们想处理C-tree, 那么数据的处理流程是:
sent--(cL/DL)-->C/D_tree---->graph_data in matrix
sent---->index_data---->embedding
it seems i need a package to solve graph_like data
# big graph has some 并行度去加速
'''

'''
并行度的角度上看, Graph内部还是有一些并行度的
如果每次一次边的inference代价为O(200)甚至O(200*200)，那么预先用O(N)的时间建立好最优的
并行计算序列是能提升速度的。
或者调用稀疏矩阵的乘法的接口

numpy可以表示任意的结构体数据。
然后inference定义为：边怎么传播计算，节点怎么发出pluse(与神经网络一样一样的...)
inference逻辑是：每次先考虑并行，然后scan
'''
class SimpleRecursive():
    def __init__(self):
        '''
        ''' 

    def apply(self, ):


class AbstractStructualInference():
    
    def __init__(self):
        '''
        想做一个抽象的接口，因为是用theano-向量去实现的，就先要把
        '''
                        
                
                
    def apply(self, structure, nodes, arcs, ops):
        #handle_tasks(structure)
        
        
        result=apply_once(structure_data, data)
        return result

def handle_tasks(structures, task_batch=100):
    '''
    structures:
    '''
    # get tasks

    # handle tasks to task_btach(one batch, one computing type)
    # every task defined as Op_type, transfrom(orifianl_locations)---> output_locations
