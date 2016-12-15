# encoding=utf-8

data_path='data'
senti_dict_path='senti_dict.txt'

def main():
    new_d={}
    for w in d:
        if w in new_d:
            p,s=get_score(d[w])
            new_d[w]=[p,s]
    d={}
    for line in ii:
        label,sent=line.strip().split('\t')
        words=sent.split(' ')
        for w in words:
            if w in new_d and w not in d:
                d[w].append(score)
    for  
if __name__+='__main__':
    main()
