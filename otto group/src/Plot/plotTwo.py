'''
Created on Mar 24, 2015

@author: Devendra
'''

import pandas as pd
import matplotlib.pyplot as plt

class PlotForTwo():
    def __init__(self, path):
        self.__path = path
        
    def plot(self, f1, f2):
        data = pd.read_csv(self.__path, header = 0)
        d1 = data[data['target'] == 'Class_1']
        d2 = data[data['target'] == 'Class_2']
        d3 = data[data['target'] == 'Class_3']
        d4 = data[data['target'] == 'Class_4']
        d5 = data[data['target'] == 'Class_5']
        d6 = data[data['target'] == 'Class_6']
        d7 = data[data['target'] == 'Class_7']
        d8 = data[data['target'] == 'Class_8']
        d9 = data[data['target'] == 'Class_9']
        plt.figure()
        plt.scatter(d1[f1], d1[f2], z = 1, c = 'b', marker='.')
        plt.scatter(d2[f1], d2[f2], c = 'g', marker='o')
        plt.scatter(d3[f1], d3[f2], c = 'r', marker='^')
        plt.scatter(d4[f1], d4[f2], c = 'c', marker='8')
        plt.scatter(d5[f1], d5[f2], c = 'm', marker='s')
        plt.scatter(d6[f1], d6[f2], c = 'y', marker='p')
        plt.scatter(d7[f1], d7[f2], c = 'k', marker='*')
        plt.scatter(d8[f1], d8[f2], c = '0.75', marker='+')
        plt.scatter(d9[f1], d9[f2], c = '0.92', marker='x')
        plt.savefig(f1 + 'vs' + f2 + '.png')
        plt.close()
    
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