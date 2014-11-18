from collections import defaultdict
from random import shuffle
import stock_downloader
import sentiment
import os
import json

class dataStore:
    def __init__(self, trainDirectory, date1, date2):
        self.sentDict = sentiment.SentimentDict()
        self.sentDict.loadSentimentsPitt("pitt_lexicon.tff")

        self.textDict = defaultdict(list)
        self.scoreDict = {}
        self.stockDict = {}
        self.fillDicts(trainDirectory, date1, date2)
        
        self.dates = list(self.stockDict.keys())
        shuffle(self.dates)
        
    def getSentiments(self):
        return self.sentDict.sentiments
    
    def getText(self):
        return self.textDict
    
    def getScores(self):
        return self.scoreDict
    
    def getStocks(self):
        return self.stockDict        
    
    def getDates(self):
        return sorted(list(self.stockDict.keys()))
    
    #fill all of the dictionaries for training
    def fillDicts(self, directory, date1, date2):
        self.fillTextAndScoreDict(directory)
        self.fillStockDict(date1, date2)
        
    #fill the text and score dicts using the training data from 2014
    def fillTextAndScoreDict(self, directory):
        for root, dirs, files in os.walk(directory):
            for f in files:
                posTotal, negTotal = 0, 0
                superPos, superNeg = 0, 0
                f = open(os.path.join(root, f), 'r')
                news = json.loads(f.read())
                if not news:
                    continue
                name = f.name.split('/')[1].strip("#")  #get the name as YYYYMMDD string
                for item in news:
                    if item['abstract']:
                        self.textDict[name].append(item['abstract'])            #append the text to a dictionary to keep all abstracts
                        pos, neg = self.sentDict.generateScore(item['abstract'])
                        posTotal += pos
                        negTotal += neg
                        if pos > 4:
                            superPos +=1
                        if neg < -4:
                            superNeg +=1
                    elif item['lead_paragraph']:
                        self.textDict[name].append(item['lead_paragraph'])            #append the text to a dictionary to keep all abstracts
                        pos, neg = self.sentDict.generateScore(item['lead_paragraph'])
                        posTotal += pos
                        negTotal += neg
                        if pos > 4:
                            superPos +=1
                        if neg < -4:
                            superNeg +=1
                self.scoreDict[name] = (posTotal, negTotal, posTotal + negTotal, superPos, superNeg)  #add a triple as the value for each date
            break
        
    def fillStockDict(self, date1, date2):
        databank = stock_downloader.stockDatabank()
        databank.download()
        removeList = []
        for date in self.textDict:
            if databank.dateList.has_key(date) and date > date1 and date < date2:
                self.stockDict[date] = (databank.getStockDirections(date), databank.getStockReturns(date)[0],databank.getStockReturns(date)[1], databank.getStockReturns(date)[2])
            else:
                removeList.append(date)
        for date in removeList:
            self.textDict.pop(date)
            self.scoreDict.pop(date)
            