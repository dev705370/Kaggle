# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:02:33 2015

@author: Devendra
"""

import pandas as pd
import numpy as np
import os
import math
from Prediction import Prediction
from os import listdir


def distance(x1,y1,x2,y2) :
    return math.sqrt((x1-x2)**2+(y1-y2)**2);
    
def samePoint(x1,y1,x2,y2) :
    if x1 == x2 and y1 == y2 :
        return True;
    return False;
    
def getRadius(x1,y1,x2,y2,x3,y3) :
    MaxRadius = 999999999999999999999999
    if samePoint(x1,y1,x2,y2) or samePoint(x2,y2,x3,y3) or samePoint(x1,y1,x3,y3):
        return -1;
    x2 -= x1;
    y2 -= y1;
    x3 -= x1;
    y3 -= y1;
    if (x2*y3) == (y2*x3) :
        return MaxRadius;
    A = np.array([[0, 0, 1], [2*x2, 2*y2, 1], [2*x3, 2*y3, 1]]);
    try :
        Ainv = np.linalg.inv(A)
        B = np.array([0, -(x2**2+y2**2), -(x3**2+y3**2)])
        C = np.dot(Ainv, B)
        return math.sqrt(C[0]**2+C[1]**2-C[2])
    except np.linalg.LinAlgError :    
        return MaxRadius;

def normalizeRadius(r) :
    if r < 1.0 :
        return r;
    else:
        return math.log(r) + 1.0

def getPathInfo(inputFile, outputFile) :
    fd = pd.read_csv(inputFile, header = 0);
    columns = ["velocity1", "velocity2", "velocity3", "acceleration", "radius"];
    pathData = pd.DataFrame(columns = columns);
    n = fd.shape[0]
    x2, y2 = fd.x[0], fd.y[0]
    x3, y3 = fd.x[1], fd.y[1]
    index = 0
    for i in range(2, n) : #change 3 to n
        x1,y1 = x2,y2
        x2,y2 = x3,y3
        x3,y3 = fd.x[i], fd.y[i]
        v1 = distance(x1,y1,x2,y2)
        v2 = distance(x2,y2,x3,y3)
        v3 = distance(x1,y1,x3,y3) / 2.0
        acc = v2-v1
        r = getRadius(x1,y1,x2,y2,x3,y3)
        if r != -1:
            r = normalizeRadius(r)
            pathData.loc[index] = [v1,v2,v3,acc,r]
            index += 1
    pathData.to_csv(outputFile, index=False);
    return pathData;

def generateForDriver(d) :
    sourcePath = "E://Kaggle//Drivers//drivers//" + str(d) + "//";
    des_path = "E://Kaggle//Drivers//PathInfo//" + str(d) + "//";
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    for i in xrange(200) : #change 1 to 200
        file_start_name = str(i+1);
        file_name = file_start_name + ".csv";
        getPathInfo(sourcePath+file_name, des_path+file_name)
#        try :
#            thread.start_new_thread(getPathInfo, (sourcePath+file_name, des_path+file_name, ));
#        except :
#            print "can't create thread"
        
        
prediction = Prediction("Parallel.csv")
path = "E://Kaggle//Drivers//drivers//";
for f in listdir(path) :
    print "running for ", f
    generateForDriver(f)
    prediction.predict(f)
#    thread.start_new_thread(generateForDriver,(i,));
#num_cores = multiprocessing.cpu_count()
#results = Parallel(n_jobs=num_cores)(delayed(generateForDriver)(i) for i in range(1,201))  