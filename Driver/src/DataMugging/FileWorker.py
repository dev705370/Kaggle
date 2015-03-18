'''
Created on Mar 5, 2015

@author: Devendra
'''

import threading

from  MathematicalModeling import MathematicalFunctions as MF
from Point import Point
import numpy as np
import pandas as pd


class FileWorkerThread(threading.Thread):
    def __init__(self, src_file, dest_file) :
        threading.Thread.__init__(self)
        self.src_file = src_file
        self.dest_file = dest_file
        
    def run(self):
        threading.Thread.run(self)
        data = pd.read_csv(self.src_file, header = 0)
        lastV = 0
        columnList = ['velocity', 'acceleration', 'angle']
        outDataArray = np.array([]).reshape((0,3)) 
        p1 = Point(data.x[0], data.y[0])
        p2 = Point(data.x[0], data.y[0])
        pointList = data.apply(lambda x : Point(x[0], x[1]), axis = 1)
        p3 = pointList.pop(0)
        for P in pointList :
            p1 = p2
            p2 = p3
            p3 = P
            v = lastV
            lastV = p2.distance(p3)
            acc = MF.getAcceleration(v, lastV)
            angle = MF.getAngle(p1, p2, p3)
            outDataArray = np.vstack((outDataArray ,[v,acc,angle]))
        outData = pd.DataFrame(outDataArray, columns= columnList)
        outData.to_csv(self.dest_file, index=False)

