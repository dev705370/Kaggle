# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 14:19:31 2015

@author: Devendra
"""

import threading
import math
import numpy as np
class DistanceThread(threading.Thread) :
    def __init__(self, x1, y1, x2, y2) :
        threading.Thread.__init__(self)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def run(self) :
        self.distance = math.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2)
        
        
class RadiusThread(threading.Thread) :
    def __init__(self, x1, y1, x2, y2, x3, y3) :
        threading.Thread.__init__(self)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        
    def samePoint(self, x1,y1,x2,y2) :
        if x1 == x2 and y1 == y2 :
            return True;
        return False;

    def run(self) :
        MaxRadius = 999999999999999999999999
        if self.samePoint(self.x1,self.y1,self.x2,self.y2) or self.samePoint(self.x2,self.y2,self.x3,self.y3) or self.samePoint(self.x1,self.y1,self.x3,self.y3):
            self.radius = -1
        x2 = self.x2 - self.x1;
        y2 = self.y2 - self.y1;
        x3 = self.x3 - self.x1;
        y3 = self.y3 - self.y1;
        if (x2*y3) == (y2*x3) :
            self.radius = MaxRadius;
        A = np.array([[0, 0, 1], [2*x2, 2*y2, 1], [2*x3, 2*y3, 1]]);
        try :
            Ainv = np.linalg.inv(A)
            B = np.array([0, -(x2**2+y2**2), -(x3**2+y3**2)])
            C = np.dot(Ainv, B)
            self.radius = math.sqrt(C[0]**2+C[1]**2-C[2])
        except np.linalg.LinAlgError :    
            self.radius = MaxRadius;