# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 16:01:48 2015

@author: Devendra
"""
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import csv
import time
from os import listdir
import matplotlib.pyplot as plt

def value(x) :
    if x > 0.9995 :
        return 1
    return 0
    
class Prediction :
    
    def __init__(self, outputFile) :
        self.outputFile = outputFile
        self.columns = ["driver_trip", "prob"]
        df = pd.DataFrame(columns = self.columns)
        df.to_csv(outputFile, index=False)
        
    def predict(self, d) :
        sourcePath = "E://Kaggle//Drivers//PathInfo3//" + str(d) + "//"
        ext = ".csv"
        data = pd.read_csv(sourcePath + str(1) + ext, header = None)
        for i in range(2, 201) :
            data = pd.concat([data, pd.read_csv(sourcePath + str(i) + ext, header = None)], ignore_index = True)
        est = KMeans(n_clusters = 2, max_iter = 1000)
        data2 = data.dropna()
        if data2.shape[0] < 2 :
            data.to_csv("test.csv", index = False)
            print "struck at ", d
            print data2
            print data
        est.fit(data2.values)
        df = pd.DataFrame(columns = self.columns) 
        for i in range(1, 201) :
            pathData = pd.read_csv(sourcePath + str(i) + ext, header = None)
            pathData = pathData.dropna()
            label = est.predict(pathData.values)
            avg = np.average(label)
            df.loc[i-1] = [str(d) + "_" + str(i), avg]
        if df.prob.min() == 0.0 :
            df.prob = df.prob.apply(lambda x : 1-x)
        plt.figure(d,figsize = (18, 10), dpi = 100)
        ax = df.plot(x='driver_trip', y = 'prob')
        fig = ax.get_figure()
        fig.savefig("//PathFigures//answer_"+str(d)+".png")
#        df.prob = df.prob.apply(lambda x : value(x))
#        with open(self.outputFile, 'ab') as fd :
#            writer = csv.writer(fd)
#            writer.writerows(df.values)  


if __name__ == "__main__" :     
    prediction = Prediction("Parallel.csv")
    path = "E://Kaggle//Drivers//drivers//";
    for f in listdir(path) :
        start_time = time.clock()
        prediction.predict(f)
        t = time.clock() - start_time
        print "completed For ",  f, " in ", t, " seconds"
