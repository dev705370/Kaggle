'''
Created on Mar 24, 2015

@author: Devendra
'''

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd

class ExtraTree():
    def __init__(self, _path, trainData, testData):
        self.__path = _path
        self.__trainData = trainData
        self.__testData = testData
        
    def runExtraTreeProb(self):
        X = self.__trainData.values
        Y = X[:, -1]
        X = X[:, :-1]
        etc = ExtraTreesClassifier(n_estimators= 1000, n_jobs = -1)
        etc.fit(X, Y)
        print 'training done'
        
        Xtest = self.__testData.ix[:,'feat_1':].values
        _id = self.__testData.ix[:,'id'].values
        _id = _id.reshape([_id.shape[0], 1])
        
        output = etc.predict_proba(Xtest)
        output = np.hstack([_id, output])
        outData = pd.DataFrame(output, columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
        outData['id'] = outData['id'].apply(self.convert)
        outData.to_csv(self.__path + '//output_ExtraTree_1000.csv', index=False)
        
    def runExtraTree(self):
        X = self.__trainData.values
        Y = X[:, -1]
        X = X[:, :-1]
        etc = ExtraTreesClassifier(n_estimators= 1000, n_jobs = -1)
        etc.fit(X, Y)
        print 'training done'
        
        Xtest = self.__testData.ix[:,'feat_1':].values
        _id = self.__testData.ix[:,'id'].values
        _id = _id.reshape([_id.shape[0], 1])
        
        output = etc.predict(Xtest)
        output = self.getProb(output)
        outData = pd.DataFrame(output, columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
        outData['id'] = outData['id'].apply(self.convert)
        outData.to_csv(self.__path + '//output_ExtraTree_1000.csv', index=False)
        
    
    def getProb(self, data):
        matrix = np.array([]).reshape([0,10])
        _id = 1
        for class_no in data :
            _row = []
            _row.append(_id)
            for i in range(1, 10) :
                if i == class_no :
                    _row.append(1)
                else :
                    _row.append(0)
            matrix = np.vstack([matrix, np.array(_row)])
            _id += 1
        return matrix    
        
    def convert(self, x):
        try:
            return x.astype(int)
        except:
            return x
        