'''
Created on Apr 29, 2015

@author: mitde05
'''

import pandas as pd
from os import listdir, path, mkdir, remove
import gzip
import zlib
from Time import date
from bokeh.glyphs import Line
import time
import multiprocessing
from joblib import Parallel, delayed

class DataFormat():
    time = 0
    system_name = 1
    job_name = 2
    metric_value = 3
    
    METRIC = 'metric'
    MIN = 'min'
    HOUR = 'hour'
    DAY = 'day'
    MONTH = 'month'
    
    
    def __init__(self, folder):
        self.folder = folder
        self.dataMap = {}
        
    def readAll(self):
#         cores = multiprocessing.cpu_count()
        cores = 6
        Parallel(n_jobs=cores)(delayed(proccessFile)(_file, self.folder) for _file in listdir(self.folder))
#         for _file in listdir(self.folder) :
#             proccessFile(_file, self.folder)
    
    def printAll(self):
        for k, v in self.dataMap.iteritems() :
            print k
            print v.shape
            v.to_csv(str(k)+'.csv', index = None)
            
    def getData(self, key):
        return self.dataMap[key].values
    
    
    
def proccessFile(_file, folder):
    start = time.clock()
    output = gzip.open(folder + _file, 'r')
    dataMap = add(output)
    saveFile(_file, dataMap)
    output.close()
    clean(_file, folder)
    print _file, ' and time = ', time.clock() - start
    
def saveFile(_file, dataMap):
    _file = _file[:-3]
    for k, v in dataMap.iteritems() :
        k = str(k)
        if not path.exists(k) :
            mkdir(k)
        v.to_csv(k + '/' + _file, index = None)

def clean(_file, folder):    
    remove(folder + _file)
    
def add(_file):
    dataMap = {}
    _line = _file.readline()
    while not _line == '':
        if not 'system_name' in _line :
            processLine(_line, dataMap)
        _line = _file.readline()
    return dataMap
        
def processLine(line, dataMap):
    line = line.strip('\n')
    values = line.split(',')
    if dataMap.has_key(values[DataFormat.system_name]) :
        data = dataMap[values[DataFormat.system_name]]
        columns = data.columns.values
        new_row = createRow(values, columns)
        data = pd.concat([data, new_row])
        data = data.fillna(0)
        dataMap[values[DataFormat.system_name]] = data
    else :
        new_row = createRow(values, None)
        dataMap[values[DataFormat.system_name]] = new_row
        
def createRow(values, columns):
    new_row = pd.DataFrame(columns=columns)
    new_row[values[DataFormat.job_name]] = [1]
    try :
        mv = float(values[DataFormat.metric_value])
    except :
        return
    new_row[DataFormat.METRIC] = [mv]
    d = date(values[DataFormat.time])
    new_row[DataFormat.MIN] = [d.getMin()]
    new_row[DataFormat.HOUR] = [d.getHour()]
    new_row[DataFormat.DAY] = [d.getDay()]
    new_row[DataFormat.MONTH] = [d.getMonth()]
    return new_row

      
if __name__ == '__main__' :
    df = DataFormat(path.dirname(__file__) + '/../../CompressData/')  
    df.readAll()
    df.printAll()