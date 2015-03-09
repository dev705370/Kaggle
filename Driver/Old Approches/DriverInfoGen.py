# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:02:33 2015

@author: Devendra
"""

import pandas as pd
import numpy as np
import math
from PredictionFromDriver import Prediction
from os import listdir
from Driver import Driver
import time


def convert(secs) :
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return str(h)+":"+str(m)+":"+str(s)

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
        return math.log(r) + 1.0

def getPathInfo(inputFile, driver) :
    fd = pd.read_csv(inputFile, header = 0);
#    columns = ["velocity1", "velocity2", "velocity3", "acceleration", "radius"];
#    pathData = pd.DataFrame(columns = columns);
    n = fd.shape[0]
    x2, y2 = fd.x[0], fd.y[0]
    x3, y3 = fd.x[1], fd.y[1]
    pathData = np.array([])
    pathData = pathData.reshape(0,5)
    for i in range(2, n) : #change 3 to n
        x1,y1 = x2,y2
        x2,y2 = x3,y3
        x3,y3 = fd.x[i], fd.y[i]
        v1 = distance(x1,y1,x2,y2)
        v2 = distance(x2,y2,x3,y3)
        v3 = distance(x1,y1,x3,y3) / 2.0
        acc = v2-v1
        r = getRadius(x1,y1,x2,y2,x3,y3)
        r = normalizeRadius(r)
        pathData = np.vstack((pathData ,[v1,v2,v3,acc,r]))
    driver.addPath(pathData)

def generateForDriver(d, driver) :
    sourcePath = "E://Kaggle//Drivers//drivers//" + str(d) + "//";
    for i in xrange(200) : #change 1 to 200
        getPathInfo(sourcePath+str(i+1)+".csv",  driver)
        
        
prediction = Prediction("WithoutDiskWrite.csv")
path = "E://Kaggle//Drivers//drivers//";
for f in listdir(path) :
    start_time = time.clock()
    driver = Driver(f)
    generateForDriver(f, driver)
    prediction.predict(driver)
    t = time.clock() - start_time
    print "completed For ",  f, " in ", t, "seconds"
