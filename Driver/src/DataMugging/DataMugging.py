'''
Created on Mar 5, 2015

@author: Devendra
'''

import multiprocessing
from os import listdir
import os
import shutil
import time

from joblib import Parallel, delayed

from FileWorker import FileWorkerThread as FWT


class DataMugging() :
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        
    def start(self) :
        cores = multiprocessing.cpu_count()
#         Parallel(n_jobs=cores)(delayed(shutil.rmtree)(self.destination + folder) for folder in listdir(self.destination))
        Parallel(n_jobs=cores)(delayed(mugFolder)(folder, self.source, self.destination) for folder in listdir(self.source))

        
def mugFolder(folder, source, destination) :
    start_time = time.clock()
    dest_folder = destination + str(folder) + "//"
    if not os.path.exists(dest_folder) :
        os.mkdir(dest_folder)
    src_folder = source + str(folder) + "//"
    workers_thread = []
    for _file in listdir(src_folder) :
        _worker = FWT(src_folder + str(_file), dest_folder + str(_file))
        _worker.start()
        workers_thread.append(_worker)
        
    for _worker in workers_thread :
        _worker.join()
    print "completed For ",  folder, " in ", time.clock() - start_time, "seconds"