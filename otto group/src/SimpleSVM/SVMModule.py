'''
Created on Mar 20, 2015

@author: Devendra
'''

from sklearn import svm, feature_extraction, cross_validation
import pandas as pd
import numpy as np
from numpy import int64

class SimpleSVM():
    def __init__(self, _path, trainData, testData):
        self.__path = _path
        self.__trainData = trainData
        self.__testData = testData
        
    def runSVM(self):
        X = self.__trainData.values
        Y = X[:, -1]
        X = X[:, :-1]
        Xtest = self.__testData.ix[:,'feat_1':].values
        _id = self.__testData.ix[:,'id'].values
        _id = _id.reshape([_id.shape[0], 1])
        
#         tfidf = feature_extraction.text.TfidfTransformer()
#         X = tfidf.fit_transform(X).toarray()
#         Xtest = tfidf.transform(Xtest).toarray()
        
        clf = svm.SVC(degree=5, probability=True, cache_size=2000)
        print 'score = ', np.mean(cross_validation.cross_val_score(clf, X, Y, n_jobs = -1))
        clf.fit(X, Y)
        print 'training done'
        output = clf.predict_proba(Xtest)
        output = np.hstack([_id, output])
        outData = pd.DataFrame(output, columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
        outData['id'] = outData['id'].apply(self.convert)
        outData.to_csv(self.__path + '//output3.csv', index=False)
        
    def convert(self, x):
        try:
            return x.astype(int)
        except:
            return x
        