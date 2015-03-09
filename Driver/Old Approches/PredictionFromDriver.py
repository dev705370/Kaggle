# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 16:01:48 2015

@author: Devendra
"""
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import csv

class Prediction :
    
    def __init__(self, outputFile) :
        self.outputFile = outputFile
        self.columns = ["driver_trip", "prob"]
        df = pd.DataFrame(columns = self.columns)
        df.to_csv(outputFile, index=False)
        
    def predict(self, driver) :
        data = driver.getAllData()
        est = KMeans(n_clusters = 2, max_iter = 10000000)
        est.fit(data)
        df = pd.DataFrame(columns = self.columns) 
        for i in range(1, 201) :
            label = est.predict(driver.getDataForPath(i))
            avg = np.average(label)
            df.loc[i-1] = [str(driver.getId()) + "_" + str(i), avg]
        if df.prob.min() == 0.0 :
            df.prob = [1-x for x in df.prob]
        with open(self.outputFile, 'ab') as fd :
            writer = csv.writer(fd)
            writer.writerows(df.values)  
        if df.prob.min() == 0.0 :
            print "Fliping not working"
