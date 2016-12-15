# encoding=utf-8

ii=open('topic.txt','r')
objects=[]
a=ii.readlines()
for i in a:
    j=i.decode('gbk').encode('utf-8')
    if len(j.split('\t'))<2:continue
    s=j.split('\t')[1]
    if '"' in s:continue
    s=s.replace(':',' ').replace('竞品','')
    for w in ['-',':','：','竞品']:
        s=s.replace(w,' ')
    o=[t.strip() for t in s.split(' ') if t!=' ' and t!='']
    objects=objects+o
print len(objects)
w=open('objects.txt','w')
w.write('\n'.join(list(set(objects))))
