#stock_downloader.py
#File that interacts with the Yahoo Finance API to download the necessary stock data for the DJIA

import csv
import urllib2
from _collections import defaultdict
import sentiment
from datetime import date, timedelta


class stockDatabank:

    def __init__(self):
        self.data = None
        self.StockDict = defaultdict(dict)
        self.dateList = {}
        self.returnList = []
        self.changeList = []

    def download(self):

        downloaded_data  = urllib2.urlopen("http://ichart.finance.yahoo.com/table.csv?s=DJIA&d=10&e=14&f=2014&g=d&a=10&b=18&c=2008&ignore=.csv")
        csv_data = reversed(list(csv.reader(downloaded_data)))
        counter = 0
        prevClose = 0
        csv_string = str(csv_data)

        localFile = open('stockdata.csv', 'w')
        for row in csv_data:
            self.dateList[row[0][0:4] + row[0][5:7] + row[0][8:]] = 1
            localFile.write(row[0] + "," + row[1] + "," + row[4] + "\n")
            if row[0] != "Date":
                self.StockDict[row[0]] = defaultdict()
                self.StockDict[row[0]]["Counter"] = counter
                self.StockDict[row[0]]["Open"] = row[1]
                self.StockDict[row[0]]["Close"] = row[4]
                if row[1] < row[4]:
                    self.changeList.append(1)
                elif row[1] > row[4]:
                    self.changeList.append(-1)
                else:
                    self.changeList.append(0)
                self.returnList.append(float(row[4]) - float(row[1]))

            counter += 1
            prevClose = row[4]
        localFile.close()

    def read(self, filename):
        datafile = open(filename, 'r')
        counter = 0
        prevClose = 0
        for line in datafile:
            pieces = line.split(',')
            if pieces[0] != "Date":
                self.StockDict[pieces[0]] = defaultdict()
                self.StockDict[pieces[0]]["Counter"] = counter
                self.StockDict[pieces[0]]["Open"] = pieces[1]
                self.StockDict[pieces[0]]["Close"] = pieces[2]
                if pieces[1] < pieces[2]:
                    self.changeList.append(1)
                elif pieces[1] > pieces[2]:
                    self.changeList.append(-1)
                else:
                    self.changeList.append(0)
                self.returnList.append(float(pieces[2]) - float(pieces[1]))
            counter += 1
            prevClose = pieces[2]

    def getStockDirections(self, date, dateTwo = None):
        #print self.StockDict[date]
        date = date[0:4] + "-" + date[4:6] + "-" + date[6:] 
        counter = self.StockDict[date]["Counter"]
        counterTwo = 0
        if dateTwo != None:
            dateTwo = dateTwo[0:4] + "-" + dateTwo[4:6] + "-" + dateTwo[6:] 
            counterTwo = self.StockDict[dateTwo]["Counter"]
            changeList = []
            for i in range(counter, counterTwo + 1):
                changeList.append(self.changeList[i])
            return changeList
        else:
            return self.changeList[counter]

    def getStockReturns(self, date, dateTwo = None):
        date = date[0:4] + "-" + date[4:6] + "-" + date[6:] 
        counter = self.StockDict[date]["Counter"]
        counterTwo = 0
        if dateTwo != None:
            dateTwo = dateTwo[0:4] + "-" + dateTwo[4:6] + "-" + dateTwo[6:] 
            counterTwo = self.StockDict[dateTwo]["Counter"]
            returnList = []
            for i in range(counter, counterTwo + 1):
                returnList.append(round(self.returnList[i], 4))
                
            return returnList
      
        else:
            return (self.returnList[counter], self.StockDict[date]["Open"], self.StockDict[date]["Close"])
    
def main():

    databank = stockDatabank()

    databank.download()

    #databank.read('./stockdata.csv')
# 
#     print databank.getStockDirections("20140102")
#     print databank.getStockReturns("20140102")