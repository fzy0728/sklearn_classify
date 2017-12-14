# coding:utf-8

import random

def readfile(filepath):
    dateset = open(filepath,'r')
    return dateset.read().split('\n')

if __name__ == '__main__':

    trainset = open('../date/trainset.txt','w+')
    for i in random.sample(set(readfile('../date/log.txt')),500):
        trainset.write(i+'\n')
    for i in random.sample(set(readfile('../date/26.txt')),500):
        trainset.write(i+'\n')
    trainset.close()

