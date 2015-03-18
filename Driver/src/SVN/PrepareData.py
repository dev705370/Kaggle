'''
Created on Mar 8, 2015

@author: Devendra
'''

import itertools
from os import listdir
import random

import numpy as np
import pandas as pd


class PrepareData() :
    def __init__(self, driver, path, otherDrivers=200, otherDrives=2):
        self.driver = driver
        self.path = path
        self.otherDrivers = otherDrivers
        self.otherDrives = otherDrives
        
    def getSVCData(self):
        _path = self.path + self.driver
        data = np.array([]).reshape([0, 3])
        for f in listdir(_path) :
            pdData = pd.read_csv(_path + "\\" + f, header=0)
            data = np.vstack([data, pdData.values])
        return data
        
    def getData(self) :
        data = self.positiveData(self.path)
        data = np.vstack([data, self.negativeData(self.path)])
        return data
        
    def positiveData(self, path):
        path += self.driver
        data = np.array([]).reshape([0, 4])
        for f in listdir(path) :
            pdData = pd.read_csv(path + "\\" + f, header=0)
            pdData['value'] = 1
            data = np.vstack([data, pdData.values])
        return data
    
    
    def negativeData(self, path) :
        _dir = listdir(path)
        n = len(_dir)
        data = np.array([]).reshape([0, 4])
        for i in xrange(self.otherDrivers) :
            d = random.randrange(0, n)
            if _dir[d] == self.driver :
                i -= 1
                continue
            tempPath = path + _dir[d] + "\\"
            for _ in itertools.repeat(None, self.otherDrives):
                p = random.randint(1, 200)
                pdData = pd.read_csv(tempPath + str(p) + ".csv", header=0)
                pdData['value'] = 0
                data = np.vstack([data, pdData.values])
        return data
