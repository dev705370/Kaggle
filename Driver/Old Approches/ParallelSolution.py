# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:02:33 2015

@author: Devendra
"""

import pandas as pd
import numpy as np
import os
import math
import multiprocessing
from joblib import Parallel, delayed
from Prediction import Prediction
from os import listdir
import threading
import time


def distance(x1,y1,x2,y2) :
    return math.sqrt((x1-x2)**2+(y1-y2)**2);
    
def samePoint(x1,y1,x2,y2) :
    if x1 == x2 and y1 == y2 :
        return True;
    return False;
    
def getRadius(x1,y1,x2,y2,x3,y3) :
    MaxRadius = 999999999999999999999999
    if samePoint(x1,y1,x2,y2) or samePoint(x2,y2,x3,y3) or samePoint(x1,y1,x3,y3):
        return 0;
    x2 -= x1;
    y2 -= y1;
    x3 -= x1;
    y3 -= y1;
    div = 4*(x2*y3 - x3*y2)
    if div == 0 :
        return MaxRadius;
    a = x2**2 + y2**2
    b = x3**2 + y3**2
    A = 2*y3*a - 2*y2*b
    B = -2*x3*a + 2*x2*b
    r = math.sqrt(A**2+B**2)
    return float(r) / float(div);

def normalizeRadius(r) :
    if r < 1.0 :
        return r;
    else:
        return math.log(r, 10) + 1.0
class PathInfoThread(threading.Thread) :
    def __init__(self, inputFile, outputFile):
        threading.Thread.__init__(self)
        self.inputFile = inputFile
        self.outputFile = outputFile
    
    def run(self) :
        getPathInfo(self.inputFile, self.outputFile)
        
        
def getPathInfo(inputFile, outputFile) :
    fd = pd.read_csv(inputFile, header = 0);
    n = fd.shape[0]
    x2, y2 = fd.x[0], fd.y[0]
    x3, y3 = fd.x[1], fd.y[1]
    pathData = np.array([]).reshape((0,5))
    for i in range(2, n) : #change 3 to n
        x1,y1 = x2,y2
        x2,y2 = x3,y3
        x3,y3 = fd.x[i], fd.y[i]
        v1 = distance(x1,y1,x2,y2)
        v2 = distance(x2,y2,x3,y3)
        v3 = distance(x1,y1,x3,y3) / 2.0
        acc = v2-v1
        r = getRadius(x1,y1,x2,y2,x3,y3)
        if r != 0:
            r = normalizeRadius(r)
        pathData = np.vstack((pathData ,[v1,v2,v3,acc,r]))
    np.savetxt(outputFile, pathData, delimiter =',')
#    pathData.to_csv(outputFile, index=False);
    
    

def generateForDriver(d) :
    start_time = time.clock()
    sourcePath = "E://Kaggle//Drivers//drivers//" + str(d) + "//";
    des_path = "E://Kaggle//Drivers//PathInfo3//" + str(d) + "//";
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    ext = ".csv"
    threads = []
    for i in range(1, 201) :
        t = PathInfoThread(sourcePath+str(i)+ext, des_path+str(i)+ext)
        t.start()
        threads.append(t)
    for t in threads :
        t.join()
    print "completed For ",  d, " in ", time.clock() - start_time, "seconds"
#    num_cores = multiprocessing.cpu_count()
#    Parallel(n_jobs=num_cores)(delayed(getPathInfo)(sourcePath+str(i)+ext, des_path+str(i)+ext) for i in range(1,201))


        
if __name__ == "__main__" :     
    prediction = Prediction("Parallel.csv")
    num_cores = multiprocessing.cpu_count()
    path = "E://Kaggle//Drivers//drivers//";
    Parallel(n_jobs=num_cores)(delayed(generateForDriver)(f) for f in listdir(path))
#    for f in listdir(path) :
#        generateForDriver(f)
    
    print "All Data generated"        
    k = 0
    for f in listdir(path) :
        prediction.predict(f)
        k += 1
        if k % 50 == 0 :
            print "done for ", k
    #    thread.start_new_thread(generateForDriver,(i,));
    #num_cores = multiprocessing.cpu_count()
    #results = Parallel(n_jobs=num_cores)(delayed(generateForDriver)(i) for i in range(1,201))  