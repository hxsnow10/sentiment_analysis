# encoding=utf-8

class MergeModel()

    def 

def conflict(x,y):

def merge(x,y):
    '''
    x是cnn的情感值,y是senti_dict的情感值.y有4种情况,>0.5,<0.5,0.5,unknown(make as 0.50007)
    当y!=0.50007,x,y矛盾的时候就选y，不矛盾就选一个折中；y=0.50007的时候就选x。
    '''


def merge_table(words, cnn_token, cnn_table, dict_table):
    cnn_table=build(sen_len,cnn_token,cnn_table)
    table=deepcopy(dict_table)
    m,n=table.shape
    for i in range(m):
        for j in range(n):
            x=cnn_table[i,j]
            y=dict_table[i,j]
            table[i][j]=merge(x,y)

    for i in range(m):
        for j in range(n):
            for k in range(i,m):
                x=table[i,j]
                y=table[k,j]
                if conflict(x,y):
                    table[i,j]=0
    for i in range(n):
        for k in range(4):
        if d[i-k-1]+table[k,i]>d[i]:
            d[i]=d[i-k]+table[k,i]
            p[i]=k
    sentis=[]
    i=n-1
    while i>=0:
        sentis=[(table[p[i],i],words[i-p[i],i+1])]+sentis
        i=i-p[i]
    return merge(sentis)
