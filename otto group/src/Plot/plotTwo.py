'''
Created on Mar 24, 2015

@author: Devendra
'''

import pandas as pd
import matplotlib.pyplot as plt
import random

class PlotForTwo():
    def __init__(self, path):
        self.__path = path
        
    def plot(self, f1, f2):
        data = pd.read_csv(self.__path, header = 0)
        plt.figure()
        
        g = 5
        
        d1 = data[data['target'] == 'Class_1']
        d1 = d1[[f1,f2]]
        d1 = d1*g
        d1[f1] = d1[f1].apply(lambda x : self.addRandom(x))
        d1[f2] = d1[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d1[f1], d1[f2], c = 'b', marker='.')
        
        d2 = data[data['target'] == 'Class_2']
        d2 = d2[[f1,f2]]
        d2 = d2*g
        d2[f1] = d2[f1].apply(lambda x : self.addRandom(x))
        d2[f2] = d2[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d2[f1], d2[f2], c = 'g', marker='o')
        
        d3 = data[data['target'] == 'Class_3']
        d3 = d3[[f1,f2]]
        d3 = d3*g
        d3[f1] = d3[f1].apply(lambda x : self.addRandom(x))
        d3[f2] = d3[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d3[f1], d3[f2], c = 'r', marker='^')
        
        d4 = data[data['target'] == 'Class_4']
        d4 = d4[[f1,f2]]
        d4 = d4*g
        d4[f1] = d4[f1].apply(lambda x : self.addRandom(x))
        d4[f2] = d4[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d4[f1], d4[f2], c = 'c', marker='8')
        
        d5 = data[data['target'] == 'Class_5']
        d5 = d5[[f1,f2]]
        d5 = d5*g
        d5[f1] = d5[f1].apply(lambda x : self.addRandom(x))
        d5[f2] = d5[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d5[f1], d5[f2], c = 'm', marker='s')
        
        d6 = data[data['target'] == 'Class_6']
        d6 = d6[[f1,f2]]
        d6 = d6*g
        d6[f1] = d6[f1].apply(lambda x : self.addRandom(x))
        d6[f2] = d6[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d6[f1], d6[f2], c = 'y', marker='p')
        
        d7 = data[data['target'] == 'Class_7']
        d7 = d7[[f1,f2]]
        d7 = d7*g
        d7[f1] = d7[f1].apply(lambda x : self.addRandom(x))
        d7[f2] = d7[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d7[f1], d7[f2], c = 'k', marker='*')
        
        d8 = data[data['target'] == 'Class_8']
        d8 = d8[[f1,f2]]
        d8 = d8*g
        d8[f1] = d8[f1].apply(lambda x : self.addRandom(x))
        d8[f2] = d8[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d8[f1], d8[f2], c = '0.75', marker='+')
        
        d9 = data[data['target'] == 'Class_9']
        d9 = d9[[f1,f2]]
        d9 = d9*g
        d9[f1] = d9[f1].apply(lambda x : self.addRandom(x))
        d9[f2] = d9[f2].apply(lambda x : self.addRandom(x))
        plt.scatter(d9[f1], d9[f2], c = '0.92', marker='x')
        
        plt.savefig(f1 + 'vs' + f2 + '.png')
        plt.show()
        plt.close()

    def addRandom(self, x):
        return x + random.uniform(-2, 2)
    
    def getColor(self, x):
        if x == 'Class_1' :
            return 'b'
        if x == 'Class_2' :
            return 'g'
        if x == 'Class_3' :
            return 'r'
        if x == 'Class_4' :
            return 'c'
        if x == 'Class_5' :
            return 'm'
        if x == 'Class_6' :
            return 'y'
        if x == 'Class_7' :
            return 'k'
        if x == 'Class_8' :
            return '0.75'
        if x == 'Class_9' :
            return '0.95'