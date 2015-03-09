'''
Created on Mar 8, 2015

@author: Devendra
'''

from os import listdir
import pandas as pd
import numpy as np
import random
import itertools

class PrepareData() :
    def __init__(self, driver, otherDrivers = 200, otherDrives = 2):
        self.driver = driver
        self.otherDrivers = otherDrivers
        self.otherDrives = otherDrives
        
    def getData(self) :
        path = 'E:\\Python\\Workspace\\Kaggle\\Driver\\resource\\DataMugging\\'
        data = self.positiveData(path)
        data = np.vstack([data, self.negativeData(path)])
        
        
    def positiveData(self, path):
        path += self.driver
        data = np.array().reshape([0, 4])
        for f in listdir(path) :
            pdData = pd.read_csv(path+"\\"+f, header = 0)
            pdData['value'] = 1
            data = np.vstack([data, pdData.values()])
        return data
    
    
    def negativeData(self, path) :
        _dir = listdir(path)
        n = len(_dir)
        data = np.array().reshape([0, 4])
        for i in xrange(self.otherDrivers) :
            d = random.randrange(0, n)
            if _dir[d] == self.driver :
                i -= 1
                continue
            tempPath = path + _dir[d] + "\\"
            for _ in itertools.repeat(None, self.otherDrives):
                p = random.randint(1, 200)
                pdData = pd.read_csv(tempPath + str(p) + ".csv", header = 0)
                pdData['value'] = 0
                data = np.vstack([data, pdData.values()])
        return data