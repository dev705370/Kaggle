# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 21:18:52 2015

@author: Devendra
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import math
from os import listdir
import time

def plotForDriver(d) :
    sourcePath = "E://Kaggle//Drivers//drivers//" + str(d) + "//";
    des_path = "E://Kaggle//Drivers//PathFigures2//" + str(d) + "//";
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    for i in xrange(200) :
        file_start_name = str(i+1);
        file_name = file_start_name + ".csv";
        imageName = file_start_name+".png";
        df = pd.read_csv(sourcePath+file_name, header = 0);
        plt.figure(i, figsize = (18, 10), dpi = 500);
        ax = df.plot(x='x', y='y');
        fig = ax.get_figure();
        fig.savefig(des_path+imageName);
#        del fig
#        del ax
#        del df
#        gc.collect()
        
def solve(data) :
#    print data.x[1].types
    data['velo'] = np.zeros(data.shape[0])
    data['acc'] = np.zeros(data.shape[0])
    for i in xrange(data.shape[0]-1) :
        data.velo[i+1] = math.sqrt((data.x[i+1] - data.x[i])**2 + (data.y[i+1] - data.y[i])**2);
    for i in xrange(data.shape[0]-2) :
        data.acc[i+2] = data.velo[i+2]-data.acc[i+1];
    print data.velo.tail(data.shape[0]-1).mean();
    print data.acc.tail(data.shape[0]-2).mean();
    
    
def plotDistance(d) :
    sourcePath = "E://Kaggle//Drivers//PathInfo2//" + str(d) + "//";
    des_path = "E://Kaggle//Drivers//PathFigures//" + str(d) + "//";
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    columns = ['pathId', 'Distance', 'avgAcc']
    data = pd.DataFrame(columns = columns)
    for i in xrange(200) :
        fileName = str(i+1)+".csv"
        df = pd.read_csv(sourcePath + fileName, header = None)
        s = df[0].sum()
        s += df.loc[0,0]
        s += df.loc[df.shape[0] - 1, 0]
        s /= 2
        s = math.log(s)
        data.loc[i] = [str(i+1), s, df[3].mean()]
    plt.figure(999999999999998, figsize = (18, 10), dpi = 100)
    ax = data.plot(x='pathId', y = 'Distance')
    fig = ax.get_figure()
    fig.savefig(des_path+"DistanceLogPlot.png")
    plt.figure(9999999999999, figsize = (18, 10), dpi = 100)
    ax2 = data.plot(x='pathId', y = 'avgAcc')
    fig2 = ax2.get_figure()
    fig2.savefig(des_path+"AccPlot.png")
    
        
if __name__ == "__main__" :
#    for f in listdir("drivers//") :
#        start_time = time.clock()
    plotForDriver(1080)    
#        print "completed For ",  f, " in ", time.clock() - start_time, "seconds"
#solve(pd.read_csv("E://Kaggle//Drivers//drivers//1//1.csv", header = 0));