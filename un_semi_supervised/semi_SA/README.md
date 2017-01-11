semi_supervised Sentiment
-------------
* COST相加
* 一个问题是在每个batch中选择supervised与unsupervised,因为batch是对总COST的切分
    那么最好一个semi_batch,蕴含k1个label，与k2个un_label，k1,k2取决于原本COST
    中2个COST的比例。cost_final=cost1+k*cost2
* 是否multi应该是可以控制的
* 

