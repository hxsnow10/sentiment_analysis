Sentence Generates
-----------------

####业务场景：
1. 针对一个话题，比如给一个主体，一个主体以及关于他的描述，一个主体以及相关的新闻资料，给出一个总结/评价。

2. 针对别人的评论，怎么回复。

在舆情场景中，我们可能指定期望的情感方向，或者其他特性来控制输出语句。

####技术
Inference, 以人为例，他需要：
* 懂语言
* 有自己的一套知识、逻辑、表示习惯
* 需要理解主体各方面的信息
* 需要理解需求/问题（情感方向，表达约束）
* 经过思考、输出、复审最后产出一篇文章。

Learning, 简单地讨论下，
* Input(object,Sentiment)+Network-->Output(Sents), 有监督学习
* 怎么把任意的情感模型/关于主体的情感模型 注入, 在COST中加入即可。在生成模型中加入一个约束。
* 最重要的问题还是怎么学会理解基础语言学、理解句子、学会思考、学会表达。粗粒度的统计学没法解决这些问题。
