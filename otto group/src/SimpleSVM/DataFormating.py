'''
Created on Mar 20, 2015

@author: Devendra
'''

import pandas as pd

class DataFormating():
    
    @staticmethod
    def formatTrainData(_path):
        data = pd.read_csv(_path + '//train.csv')
        data['class'] = map(lambda x: DataFormating.f(x), data.target)
        data = data.drop(['id', 'target'], axis = 1)
        return data
    
    @staticmethod
    def f(x):
        return int(x[-1])
    
    @staticmethod
    def formatTestData(_path):
        data = pd.read_csv(_path + '//test.csv')
        return data
        