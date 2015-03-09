'''
Created on Mar 5, 2015

@author: Devendra
'''
import math


class Point() :
    def __init__(self, x, y):
        self.x = x;
        self.y = y
        
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __isub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other):
        return self.x != other.x or self.y == other.y
    
    def mod(self) :
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def __mul__(self, other):
        return self.x * other.x + self.y * other.y
    
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
    def __len__(self):
        return 2
