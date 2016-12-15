# encoding=utf-8
d=[]

# 中文汉字
for ch in xrange(0x4e00, 0x9fa6):
    d.append(unichr(ch).encode('utf-8'))

# 中文标点符号
cf='。？！，、；：’‘“”（）【】《》()[]{}'
cf=[i.encode('utf-8') for i in cf.decode('utf-8')]
cf.append(unichr(0x2014).encode('utf-8'))#破折号
cf.append(unichr(0x2026).encode('utf-8'))#省略号

d=d+cf

# 英文字母数字
d=d+[chr(i).encode('utf-8') for i in range(32,126) if chr(i).encode('utf-8') not in d]

oo=open('chara.txt','w')
for i in d:
    oo.write(i+'\n')
