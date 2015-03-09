'''
Created on Mar 5, 2015

@author: Devendra
'''
import math


class MathematicalFunctions() :
    @classmethod
    def getAcceleration(cls, v1, v2):
        return v2 - v1
    
    @classmethod
    def getAngle(cls, p1, p2, p3):
        if p1 == p2 or p2 == p3 :
            return 0.0
        vector1 = p2 - p1
        vector2 = p3 - p2
        value = (vector1 * vector2) / (vector1.mod() * vector2.mod())
        if value > 1.0 :
            value = 1.0
        if value < -1.0 :
            value = -1.0
        try :
            return math.acos(value)
        except ValueError :
            print value