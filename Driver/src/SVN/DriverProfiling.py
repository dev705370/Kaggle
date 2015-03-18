'''
Created on Mar 9, 2015

@author: Devendra
'''
from os import listdir
import os
import time

from sklearn import svm

from PrepareData import PrepareData
# import matplotlib.pyplot as plt?
import pandas as pd

import numpy as np


class DriverProfile():
    def __init__(self, _path, driver):
        self.path = _path
        self.driver = driver
        
    def run(self):
        PD = PrepareData(self.driver, self.path)
        data = PD.getSVCData()
        clf = svm.OneClassSVM(nu=0.15, cache_size=200, kernel='poly')
        clf.fit(data)
        dest_folder = 'E://Python//Workspace//Kaggle//Driver//resource//Profile//'
        if not os.path.exists(dest_folder) :
            os.mkdir(dest_folder)
        i = 1
        array = np.array([]).reshape([0,2])
        for _file in listdir(self.path + self.driver) :
            testData = pd.read_csv(self.path + self.driver + '\\' + _file, header=0)
            y = clf.predict(testData.values)
            a = 1
            if sum(y) < 0:
                a = 0
            array = np.vstack([array, [self.driver + '_' + str(i), a]])
            i += 1
        result = pd.DataFrame(data=array, columns=['id', 'value'])
        result.to_csv(dest_folder + self.driver + ".csv", index=False)