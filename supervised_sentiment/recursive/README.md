

####Papers
https://scholar.google.com.hk/scholar?cites=14795161263853850797&as_sdt=2005&sciodt=0,5&hl=zh-TW

####structureRNN
* 最经典的structure包括 线性表、树、图
* 传统上，我们就有关于面向图的许多算法。
* 神经网络本身就是图
* 概率图也是很经典的关于图的算法
* 这里我们要考虑的是：图与向量计算。
* 语法树(上下文无关语法)的解析与利用，或者上下文有关语法(复杂的symbolic符号系统)，怎么用向量计算来表示？

####Implementation
* 一个DL经典的recursive的拓展，即每个node用一些张量表示，每个(有向)边与node映射计算，Inference从一定的初始化
    顺着网络计算，直到某些节点得到结果，（如果有有向环，就涉及到了动力学)
* 我期望一般的RNN能实现任意的图或者树的RNN的等同功能。
* 上层逻辑：每个node与arc用numpy strucuarl array存储，定义2种算子，node怎么pluse,边怎么传播，与2种数据独立
  然后每次传入bacth的nodes, arcs, structure, ops，然后首先生成一个关于task_batch的新的向量，每一列表示同一种type
  的具体任务（他的optyepe,参数位置s,输出位置s), 然后scan批量操作，每次操作根据位置从结果矩阵R里读取，结果写入R
