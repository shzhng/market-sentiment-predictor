import csv
import urllib2
from _collections import defaultdict

class stockDatabank:

    def __init__(self):
        self.data = None
        self.StockDict = defaultdict(dict)
        self.returnList = []
        self.changeList = []

    def download(self):

        downloaded_data  = urllib2.urlopen("http://ichart.finance.yahoo.com/table.csv?s=DJIA&d=10&e=14&f=2014&g=d&a=0&b=1&c=2000&ignore=.csv")
        csv_data = reversed(list(csv.reader(downloaded_data)))
        counter = 0
        prevClose = 0
        csv_string = str(csv_data)

        localFile = open('file.csv', 'w')
        for row in csv_data:
            localFile.write(row[0] + "," + row[1] + "," + row[4] + "\n")
            if row[0] != "Date":
                self.StockDict[row[0]] = defaultdict()
                self.StockDict[row[0]]["Counter"] = counter
                self.StockDict[row[0]]["Open"] = row[1]
                self.StockDict[row[0]]["Close"] = row[4]
                if counter != 0:
                    if prevClose < row[4]:
                        self.changeList.append(1)
                    elif prevClose > row[4]:
                        self.changeList.append(-1)
                    else:
                        self.changeList.append(0)
                    self.returnList.append(float(row[4]) - float(prevClose))
                else:
                    self.changeList.append(-9999)
                    self.returnList.append(-9999)
            counter += 1
            prevClose = row[4]

        localFile.close()

    def getChange(self, date, dateTwo = None):
        counter = self.StockDict[date]["Counter"]
        counterTwo = 0

        if dateTwo != None:
            counterTwo = self.StockDict[dateTwo]["Counter"]
            changeList = []
            for i in range(counter + 1, counterTwo + 1):
                changeList.append(self.changeList[i])
            return changeList
        else:
            return self.changeList[counter]

    def getReturn(self, date, dateTwo = None):
        counter = self.StockDict[date]["Counter"]
        counterTwo = 0

        if dateTwo != None:
            counterTwo = self.StockDict[dateTwo]["Counter"]
            overallReturn = 0.0
            for i in range(counter + 1, counterTwo + 1):
                overallReturn += self.returnList[i]
            return overallReturn
        else:
            return self.returnList[counter]


def main():

    databank = stockDatabank()
    databank.download()

    print databank.getChange("2004-01-02", "2004-01-08")
    print databank.getReturn("2004-01-02", "2004-01-08")


main()
