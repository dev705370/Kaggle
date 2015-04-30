'''
Created on Apr 29, 2015

@author: Devendra
'''

import pandas as pd

class DataHolder():
    def __init__(self, columns):
        self.data = pd.DataFrame(columns=columns)
        
    def add(self, d):
        self.data = pd.concat([self.data, d])