# coding:utf-8

from sklearn.svm import SVC
from sklearn import linear_model

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from util import *
import util
import json

class model:
    def __init__(self):
        self.ridden_vocablist = []

    def getfeature(self):
        #粗筛的数据存放在trainset.txt
        dateset = util.readfile('../date/trainset.txt')
        stopword = util.loadstopword()
        vocablist = util.cutword(dateset, stopword)
        dic = util.vocfeature(util.bankofword(vocablist))
        cvocablist = []
        for key, value in dic[:100]:
            cvocablist.append(key)
        self.ridden_vocablist = cvocablist
        traindateset = util.cutlist(dateset, stopword)

        trainset = []
        for seq_list in traindateset:
            trainset.append(util.wordtovec(cvocablist, seq_list))
        self.ridden_X = np.array(trainset)
        self.ridden_y = np.append(np.ones(500), np.zeros(500))
        #精筛的数据存放在250train.xlsx
        train_from_xlsx = pd.read_excel('../date/413train.xlsx')

        traindate = list(train_from_xlsx.get('text'))

        vocablist = cutword(traindate, stopword)

        dic = vocfeature(bankofword(vocablist))

        cvocablist = []

        for key, value in dic[:300]:
            cvocablist.append(key)
        self.fine_vocablist = cvocablist
        traindateset = cutlist(traindate, stopword)

        trainset = []
        for seq_list in traindateset:
            trainset.append(wordtovec(cvocablist, seq_list))
        self.fine_X = np.array(trainset)
        self.fine_Y = np.array(train_from_xlsx.get('intent'))

    def trainlr(self):
        #lr
        lr1 = linear_model.LogisticRegression(C=10)
        lr1.fit(self.ridden_X,self.ridden_y)
        joblib.dump(lr1, "../model/ridden_train_model.m")

        #svm
        # lr1 = linear_model.LogisticRegression(C=10)
        # lr1.fit(self.ridden_X,self.ridden_y)
        # joblib.dump(lr1, "../model/ridden_train_model.m")

        #lr
        lr2 = linear_model.LogisticRegression(C=10)
        lr2.fit(self.fine_X, self.fine_Y)
        joblib.dump(lr2, "../model/fine_train_model.m")

        # # svm
        # lr2 = linear_model.LogisticRegression(C=10)
        # lr2.fit(self.fine_X, self.fine_Y)
        # joblib.dump(lr2, "../model/fine_train_model.m")

    def parse(self,text):
        answers = {}
        cls = joblib.load("../model/ridden_train_model.m")

        text_vec =  np.array([wordtovec(self.ridden_vocablist,text)])
        if(0==cls.predict(text_vec)[0]):
            answers['answer'] = "非请假-0："
            answers['result_id'] = 0
            print "非请假-0：", text
            return answers
        else:
            cls = joblib.load("../model/fine_train_model.m")
            text_vec = np.array([wordtovec(self.fine_vocablist,text)])
            if(0==cls.predict(text_vec)[0]):
                answers['answer'] = "非请假-1："
                answers['result_id'] = 0
                print '非请假-1：', text
                return answers
            else:
                answers['answer'] = "请假     "
                answers['result_id'] = 1
                print '请假    ：', text
                return answers


    def crossvad(self):
        # # Logistic Regression
        lr = linear_model.LogisticRegression(C=10)
        # lr.fit(X_train, y_train)
        # print('accuracy: {}'.format(sum(lr.predict(X_test) == y_test) / len(y_test)))
        scores = cross_val_score(lr, self.ridden_X, self.ridden_y, cv=5)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #
        # # SVM
        # clf = SVC(C=1.0, kernel='linear', decision_function_shape='ovo', probability=True)
        # # clf.fit(X_train, y_train)
        # # print('accuracy: {}'.format(sum(clf.predict(X_test) == y_test) / len(y_test)))
        # # cross validation
        # scores = cross_val_score(clf, X, Y, cv=5)
        # print(scores)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def crossvaf(self):
        # # Logistic Regression
        lr = linear_model.LogisticRegression(C=10)
        # lr.fit(X_train, y_train)
        # print('accuracy: {}'.format(sum(lr.predict(X_test) == y_test) / len(y_test)))
        scores = cross_val_score(lr, self.fine_X, self.fine_Y, cv=5)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # # SVM
        # clf = SVC(C=1.0, kernel='linear', decision_function_shape='ovo', probability=True)
        # # clf.fit(X_train, y_train)
        # # print('accuracy: {}'.format(sum(clf.predict(X_test) == y_test) / len(y_test)))
        # # cross validation
        # scores = cross_val_score(clf, self.fine_X, self.fine_Y, cv=5)
        # print(scores)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__=='__main__':
    ls = model()
    ls.getfeature()
    ls.trainlr()
    ls.crossvad()
    ls.crossvaf()


    print "-------------粗筛-------------"
    ls.parse(u'我想看刘德华电影')
    ls.parse(u'我想让我老婆帮我带饭')
    ls.parse(u'来一个鱼香肉丝')
    ls.parse(u'丽丽姐,帮我要个鱼香肉丝')
    ls.parse(u'你知道吗听说王总要结婚了')
    ls.parse(u'我想听歌曲南征北战')
    ls.parse(u'怎么没人来啊')
    ls.parse(u'你知道吗,快乐大本营,就是哪个谁,谢娜,怀孕了')
    ls.parse(u'还有这么坑爹,他有什么权利')
    ls.parse(u'9+9等于多少我都能忘,我真是醉了')

    print "-------------请假--------------"
    ls.parse(u'我要请假一天')
    ls.parse(u'可以请假吗')
    ls.parse(u'老板，明天我孩子高考，想请个假')
    ls.parse(u'小张，我明天家里有点事情，我得请假')
    ls.parse(u'李总，我肚子疼，我想请半天假')
    ls.parse(u'我今天下午来不了了，帮我请个假')
    ls.parse(u'我明天休假，工作靠你了')
    ls.parse(u'我想请三天年假')
    ls.parse(u'我要调休一天')
    ls.parse(u'我好难受,请假一天,我明天上班')
    ls.parse(u'李总,我帮丽丽请个假,她家人来北京了')


    print "------------像请假,但是不是------"
    ls.parse(u'我之前请了多少天假了帮我查一下')
    ls.parse(u'我之前请假了几天？我还能不能请假了？')
    ls.parse(u'昨天请假了，没来上班，今天有什么事情吗？')
    ls.parse(u'我老婆这几天休年假')
    ls.parse(u'我的假期还没用完吧')
    ls.parse(u'李总，我前天感冒了，请了个假，十分抱歉')
    ls.parse(u'假如明天你有空，我请你吃饭')
    ls.parse(u'请问,假设我还活着')
    ls.parse(u'别说了,我今天事情太多,你的事情我交给老吴了,')
    ls.parse(u'请问,明天可以帮我带个面包吗')