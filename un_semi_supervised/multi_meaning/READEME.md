Multi Meaning
-----------------
Targeted at training multi-meaning word vectors.

There are many methods/tricks :
1. cluster_based: first train a word2vec model, then use context average vector to clutser 
    multi meaning of center word(M-M Model).
2. M-M skip-gram model.first predict meaning of center word(encoder), then predict context word 
    from this center meaning(decoder).
3. Multi-Language based
4. Dict based

目前是每个方法一个文件夹，更好的是把他们放到一个接口里，用参数去控制。
