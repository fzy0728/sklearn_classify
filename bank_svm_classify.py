# coding:utf-8

from sklearn.svm import SVC
from sklearn import linear_model

from sklearn.model_selection import cross_val_score
import pandas as pd
import jieba as jb
import numpy as np
import re

def readfile(filepath):
    dateset = open(filepath,'r')
    return dateset.read().split('\n')

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
def riddletrain():
    dateset = readfile('../date/trainset.txt')
    stopword = loadstopword()
    vocablist = cutword(dateset,stopword)
    dic = vocfeature(bankofword(vocablist))
    #选取前n特征
    cvocablist = []
    for key,value in dic[:100]:
        cvocablist.append(key)

    traindateset = cutlist(dateset,stopword)

    print ','.join(cvocablist)
    print ','.join(cutlist(dateset,stopword)[0])
    trainset = []
    for seq_list in traindateset:
        trainset.append(wordtovec(cvocablist,seq_list))
    X = np.array(trainset)
    Y = np.append(np.ones(500),np.zeros(500))
    #print Y

    # Logistic Regression
    lr = linear_model.LogisticRegression(C=10)
    # lr.fit(X_train, y_train)
    # print('accuracy: {}'.format(sum(lr.predict(X_test) == y_test) / len(y_test)))
    scores = cross_val_score(lr, X, Y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # SVM
    clf = SVC(C=1.0, kernel='linear', decision_function_shape='ovo', probability=True)
    # clf.fit(X_train, y_train)
    # print('accuracy: {}'.format(sum(clf.predict(X_test) == y_test) / len(y_test)))
    # cross validation
    scores = cross_val_score(clf, X, Y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



def finetrain():
    train_from_xlsx = pd.read_excel('../date/250train.xlsx')

    traindate = list(train_from_xlsx.get('text'))

    Y = np.array(train_from_xlsx.get('intent'))

    stopword = loadstopword()

    vocablist = cutword(traindate, stopword)

    dic = vocfeature(bankofword(vocablist))

    cvocablist = []
    for key, value in dic[:2000]:
        cvocablist.append(key)

    traindateset = cutlist(traindate, stopword)

    print ','.join(cvocablist)
    print ','.join(cutlist(traindate, stopword)[0])
    trainset = []
    for seq_list in traindateset:
        trainset.append(wordtovec(cvocablist, seq_list))
    X = np.array(trainset)

    # Logistic Regression
    lr = linear_model.LogisticRegression(C=10)
    # lr.fit(X_train, y_train)
    # print('accuracy: {}'.format(sum(lr.predict(X_test) == y_test) / len(y_test)))
    scores = cross_val_score(lr, X, Y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # SVM
    clf = SVC(C=1.0, kernel='linear', decision_function_shape='ovo', probability=True)
    # clf.fit(X_train, y_train)
    # print('accuracy: {}'.format(sum(clf.predict(X_test) == y_test) / len(y_test)))
    # cross validation
    scores = cross_val_score(clf, X, Y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__=='__main__':
    riddletrain()
    finetrain()