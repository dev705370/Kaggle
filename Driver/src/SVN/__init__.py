from os import listdir

from DriverProfiling import DriverProfile

import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed
from SVN.svnRegression import DriverProfileRegression

def fun1(path, folder):
    start_time = time.clock()
    DP = DriverProfile(path, folder)
    DP.run()  
    print "Completed for ", folder, " in ", time.clock() - start_time, "seconds"
    
def fun2(path, folder):
    start_time = time.clock()
    DP = DriverProfileRegression(path, folder)
    DP.run()  
    print "Completed for ", folder, " in ", time.clock() - start_time, "seconds"

if __name__ == '__main__' :
    path = 'E:\\Python\\Workspace\\Kaggle\\Driver\\resource\\DataMugging1\\'
#     for folder in listdir(path) :
#         fun(path, folder)
    cores = multiprocessing.cpu_count()
    Parallel(n_jobs=cores)(delayed(fun1)(path, folder) for folder in listdir(path))
#     fun2(path, '1')