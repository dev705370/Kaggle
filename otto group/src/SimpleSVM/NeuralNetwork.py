'''
Created on Mar 27, 2015

@author: Devendra
'''

from BackPrapogation.backPrapogation import BackPrapogation
import numpy as np
import pandas as pd

class ANN():
    def __init__(self, _path, trainData, testData):
        self.__path = _path
        self.__trainData = trainData
        self.__testData = testData
        
    def getProba(self, X):
        matrix = np.array([]).reshape([0,9])
        for class_no in X :
            _row = []
            for i in range(1, 10) :
                if i == class_no :
                    _row.append(1)
                else :
                    _row.append(0)
            matrix = np.vstack([matrix, np.array(_row)])
        return matrix    
        
    def run(self):
        X = self.__trainData.values
        Y = X[:, -1]
        X = X[:, :-1]
        Y = self.getProba(Y)
        Xtest = self.__testData.ix[:,'feat_1':].values
        _id = self.__testData.ix[:,'id'].values
        _id = _id.reshape([_id.shape[0], 1])
        
        bp = BackPrapogation(Layers = 2, LayerSize = 100, learningRate=0.1, minError = 1)
        bp.fit(X, Y)
        output = bp.predict(Xtest)
        
        output = np.hstack([_id, output])
        outData = pd.DataFrame(output, columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
        outData['id'] = outData['id'].apply(self.convert)
        outData.to_csv(self.__path + '//nnSubmission1.csv', index=False)
    
    def convert(self, x):
        try:
            return x.astype(int)
        except:
            return x
        

from DataFormating import DataFormating as DF
import time

if __name__ == '__main__' :
    _path = 'E://Python//Workspace//KaggleData//otto group'
    start = time.clock()
    trainData = DF.formatTrainData(_path)
    testData = DF.formatTestData(_path)
    print 'Data formatted'
    nn = ANN(_path, trainData, testData)
    nn.run()
    print 'run time is ', time.clock() - start, 'secs'
    