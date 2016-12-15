# encoding=utf-8
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


rr='$'
rr2='。'
def clean(WeiboContent):
    content=WeiboContent
    l_clean=[]
    #l_clean.append("回复@[\w\W]+?:")    #reply someone in the front of the content
    l_clean.append(u"//@.+")       #微博回复都去掉
    l_clean.append(u"@[\w\W]+? ")         #@去掉
    l_clean.append(u"http://[0-9A-Za-z\./]*")
    l_clean.append(u"来自 ?[\S]+[ ）)。/]?")
    l_clean.append(u"via ?[\S]+[ ）)。/]?")
    l_clean.append(u"分享自 ?[\S]+[ ）)。/]?")
    l_clean.append(u"根?据 ?[\S]+消息")
    l_clean.append(u"微信：[0-9A-Za-z]*")
    l_clean.append(u"[vV]信：[0-9A-Za-z]*")
    l_clean.append(u"[0-9]{11}")
    for i in l_clean:
        if i==u"http://[0-9A-Za-z\./]*" or i==u"//@.+" or i==u"via ?[\S]+[ ）)。/]?" or \
            i==u"来自 ?[\S]+[ ）)。/]?" or i==u"分享自 ?[\S]+[ ）)。/]?":ss=rr
        else:
            ss=rr2
        (content,num)=re.subn(i,ss,content)
        #print content
    return content

def pre_weibo2(text):
    text=text.decode('utf-8')
    text = re.sub(u'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text )
    return text.encode('utf-8')

def pre_weibo(text, istitle= False):
    text=text.decode('utf-8')
    # remove strip simbols
    black_simbols=[u'　',u' ', u'\n']
    while len(text)>=1 and text[0] in black_simbols:
        text=text[1:]
    while len(text)>=1 and text[-1] in black_simbols:
        text=text[:-1]
    text = text.replace('&gt;',rr)
    text = re.sub(u'\(\S+\)',rr,text)
    print '() pred',text
    # remove url
    text = re.sub(u'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', rr2, text )
    print 'url pred',text
    # remove weibo@
    #text=clean(text)
    print 'weibo_clean pred',text
    text = re.sub(u'@[\u4e00-\u9fa5\w]+', rr, text )
    print '@ pred',text
    # number TODO some muber like 1314,213 shouldn't be removed
    text=re.sub(u'(?!)([0-9]{2,4}年)?([0-9]{1,2}月)?([0-9]{1,2}日)?([0-9]{1,2}[:：][0-9]{1,2})?',rr,text) #处理日期+数字
    print '日期 pred',text
    #text=re.sub(u'[A-Za-z]+-?[0-9]+',rr,text)# 处理TB-230这样的产品型号
    #print '型号 pred',text
    #text=re.sub(u'\d+(\.\d+)?[%]?', rr, text)
    #print '数字 pred',text
    text=re.sub(u'[一二三四五六七八九0-9]毛[一二三四五六七八九0-9]?',rr,text)
    print '钱 pred',text
    text = re.sub(u'（[\S ]+?）',rr,text) 
    text = re.sub(u'<[\S ]+?>',rr,text) 
    text = re.sub(u'\([\S ]+?\)',rr,text) 
    text = re.sub(u'#[\S ]+?#',rr2,text)
    text = re.sub(u'!!',rr,text) 
    utext=text
    if text.count(u' ')*1.0/(len(text)+0.0001)>=0.4:
        text=utext.replace(u' ','')
    if text.count(u'　')*1.0/(len(text)+0.0001)>=0.4:
        text=utext.replace(u'　','')
    text = re.sub(u' ',u',',text) 
    text = re.sub(u'　',u',',text) 
        
    '''
    text=text.replace(u'(','')
    text=text.replace(u')','')
    text=text.replace(u'（','')
    text=text.replace(u'）','')
    text=text.replace(u'【','')
    text=text.replace(u'】','')
    '''
    return text.encode('utf-8')

if __name__=='__main__':
    x=raw_input('Input:')
    print pre_weibo(x)
