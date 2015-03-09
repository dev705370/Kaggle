# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:07:04 2015

@author: Devendra
"""

import numpy as np

class Driver :
    def __init__(self, driver) :
        self.id = driver
        self.list = []
        
    def addPath(self, data) :
        self.list.append(data)
        
    def getDataForPath(self, pathId) :
        return self.list[pathId - 1]
        
    def getAllData(self) :
        data = self.list[0]
        for i in range(1, 200) :
            data = np.vstack((data, self.list[i]))
        return data
        
    def getId(self) :
        return self.id