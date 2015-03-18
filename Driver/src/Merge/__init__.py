import numpy as np
import pandas as pd
from os import listdir
import os

if __name__ == '__main__' :
    _path = 'E://Python//Workspace//Kaggle//Driver//resource//Profile//'
    dataNP = np.array([]).reshape([0,2])
    for _file in listdir(_path) :
        dataPD = pd.read_csv(_path + _file, header = 0)
        dataNP = np.vstack([dataNP, dataPD.values])
        print 'd'
    finalData = pd.DataFrame(data= dataNP, columns=['driver_trip', 'prob'])
    finalData = finalData.sort(['driver_trip'])
    finalData.to_csv('output.csv', index = False)