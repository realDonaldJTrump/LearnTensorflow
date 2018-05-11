import re
import os

pat = re.compile(r'''
                    /\*   #匹配开始的/*
                    .*?   #匹配出现任意次的任意字符,包括换行符
                    \*/   #匹配结束的*/
                    ''', re.VERBOSE|re.S)
with open('/home/gaolei/Code/LearnPython/TestRE.cpp') as f:
    text = f.read()
    #result = pat.sub('', text)
    result = pat.split(text)
    print(result)
    f.close()

lines = ['abcd', 'efgh', 'hijk']
lines = [line+os.linesep for line in lines]
with open('/home/gaolei/Code/LearnPython/test1', 'a') as f:
    f.writelines(lines)
    f.close()