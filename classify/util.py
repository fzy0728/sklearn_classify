# coding:utf-8

import jieba as jb
import numpy as np
import pandas as pd
import re
import random

def readfile(filepath):
    dateset = open(filepath,'r')
    data = dateset.read().split('\n')
    #random.shuffle(data)
    return data

def cutword(listdate,stopword):
    strinfo = re.compile(' ')
    lifeWord = []
    for i in listdate:
        for seg_word in jb.cut(strinfo.sub('',i)) :
            if(seg_word not in stopword):
                lifeWord.append(seg_word)
    return lifeWord

#统计词频
def bankofword(vocablist):
    dic = {}
    for i in vocablist:
        if(dic.has_key(i)):
            dic[i]+=1
        else:
            dic[i] = 1
    return dic
#特征提取
def vocfeature(dic):
    return sorted(dic.iteritems(), key=lambda asd: asd[1], reverse=True)

def loadstopword():
    stop = [line.strip().decode('utf-8') for line in open('../date/stopwords.txt').readlines() ]
    return stop

def cutlist(dateset,stopword):
    strinfo = re.compile(' ')
    setlist = []
    lifeword = []
    for sub_list in dateset:
        lifeword = []
        for i in jb.cut(strinfo.sub('',sub_list)):
            if (i not in stopword):
                lifeword.append(i)
        setlist.append(lifeword)
    return setlist

#bank of word生成向量
def wordtovec(cvocablist,datelist):
    seq = np.zeros(len(cvocablist) + 1)
    for i in datelist:
        if (i in cvocablist):
            seq[cvocablist.index(i)] += 1
        else:
            seq[len(cvocablist)] += 1
    return seq