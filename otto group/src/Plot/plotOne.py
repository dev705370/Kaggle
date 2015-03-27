'''
Created on Mar 25, 2015

@author: Devendra
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class PlotOne():
    
    def __init__(self, _path, _file):
        self.__path = _path
        self.__file = _file
        
    def plotAll(self):
        for i in range(1, 94) :
            self.plot(i)

    def plot(self, i):
        data = pd.read_csv(self.__path + "//" + self.__file, header = 0)
        le = LabelEncoder()
        le.fit(data['target'].values)
        data['values'] = le.transform(data['target'].values)
        plt.figure()
        x = data.iloc[:, i].values.tolist()
        y = data['values'].values.tolist()
        plt.scatter(x = x, y = y, s=5, c = 'r')
        plt.savefig("E://Python//Workspace//KaggleData//otto group//fig//" + str(i) + ".png")
        plt.close()
        