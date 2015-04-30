# import matplotlib.pyplot as plt
import pandas as pd
import os
from pylab import *

FILE1 = 'sap1training.csv'
FILE2 = 'tso1training.csv'
FILE3 = 'tso29training.csv'

if __name__ == '__main__' :
    data = pd.read_csv(os.path.dirname(__file__) + '/../../Data/' + FILE3, header = 0)
    columns = list(data.columns)
    data = data.dropna()
    data = data.drop(['cl_time'], axis=1)
    val = data[:100].values
    print val.shape
    figure(1)
    matshow(val, 1)
#     imshow(val, interpolation='bicubic')
#     grid(True)
#     savefig(os.path.dirname(__file__) + '/../../Data/1_1.png')
    show()