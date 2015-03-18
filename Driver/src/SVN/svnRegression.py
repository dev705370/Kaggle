'''
Created on Mar 12, 2015

@author: Devendra
'''

'''
Created on Mar 9, 2015

@author: Devendra
'''
from os import listdir
import os
import time

from sklearn import svm

from PrepareData import PrepareData
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np


class DriverProfileRegression():
    def __init__(self, _path, driver):
        self.path = _path
        self.driver = driver
        
    def run(self):
        PD = PrepareData(self.driver, self.path)
        data = PD.getData()
        X = data[:, :-1]
        Y = data[:, -1]
        print 'training started'
        startTime = time.clock()
        clf = svm.SVR(cache_size=1000, kernel='linear', max_iter=1000)
        clf.fit(X, Y)
        print 'traing completed in ', time.clock() - startTime, 'secs'
        dest_folder = 'E://Python//Workspace//Kaggle//Driver//resource//ProfileReg//'
        if not os.path.exists(dest_folder) :
            os.mkdir(dest_folder)
        i = 1
        array = np.array([]).reshape([0,2])
        for _file in listdir(self.path + self.driver) :
            testData = pd.read_csv(self.path + self.driver + '\\' + _file, header=0)
            y = clf.predict(testData.values)
            plt.figure(i)
            plt.plot(y)
            plt.savefig(dest_folder + 'imgae//' + str(i) + '.png')
            plt.close() 
            i+=1
#             a = 1
#             if sum(y) < 0:
#                 a = 0
#             array = np.vstack([array, [self.driver + '_' + str(i), a]])
#             i += 1
#         result = pd.DataFrame(data=array, columns=['id', 'value'])
#         result.to_csv(dest_folder + self.driver + ".csv", index=False)