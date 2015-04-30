'''
Created on Apr 29, 2015

@author: mitde05
'''

class date():
    def __init__(self, time):
        values = time.split(' ')
        self.date = values[0].split('-')
        self.watch = values[1].split(':')
        
    def getMin(self):
        return int(self.watch[1])   
    
    def getHour(self):
        return int(self.watch[0]) 
    
    def getDay(self):
        return int(self.date[2]) 
    
    def getMonth(self):
        return int(self.date[1])  