unsupervised multi-prototype
-------------------

###Introduction
最简单的方法：先训练word2vec, 然后根据每个词附近的平均word2vec不同对词的不同语义聚类，对文本词标注语义下标，然后重新训练词向量。
