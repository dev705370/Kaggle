# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 16:01:48 2015

@author: Devendra
"""
import pandas as pd
import csv
import math
from os import listdir
import time
import numpy as np

class Prediction :
    
    def __init__(self, outputFile) :
        self.outputFile = outputFile
        self.columns = ["driver_trip", "prob"]
        df = pd.DataFrame(columns = self.columns)
        df.to_csv(outputFile, index=False)
        
    def score(self, centre, data) :
        d = sum((centre - data)**2)
        return 1.0 / (math.sqrt(d) + 1.0)
        
    def predict(self, d) :
        sourcePath = "E://Kaggle//Drivers//PathInfo2//" + str(d) + "//"
        ext = ".csv"
        data = pd.read_csv(sourcePath + str(1) + ext, header = None)
        for i in range(2, 201) :
            data = pd.concat([data, pd.read_csv(sourcePath + str(i) + ext, header = None)], ignore_index = True)
        centre = [data[0].mean(), data[1].mean(), data[2].mean(), data[3].mean(), data[4].mean()]
        for i in range(1, 201) :
            pathData = pd.read_csv(sourcePath + str(i) + ext, header = None)
            pathData = (centre - pathData)**2
            pathData = pathData.sum(axis = 1)
            pathData = pathData.apply(np.sqrt) + 1
            a = pathData.max()
            avg = ((a - pathData) / a).mean()
            if avg > 0.4 :
                avg = 1
            else:
                avg = 0
            with open(self.outputFile, 'ab') as fd :
                writer = csv.writer(fd)
                writer.writerow([str(d) + "_" + str(i), avg])  
            
    
def convert(secs) :
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return str(h)+":"+str(m)+":"+str(s)

if __name__ == "__main__" :
    prediction = Prediction("Centroid2.csv")
    path = "E://Kaggle//Drivers//drivers//";
    for f in listdir(path) :
        start_time = time.clock()
        prediction.predict(f)
        t = time.clock() - start_time
        print "completed For ",  f, " in ", t, " seconds"