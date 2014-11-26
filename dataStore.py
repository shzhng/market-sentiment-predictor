#dataStore.py
#
#@author: Joshua Lang, Shuo Zheng, Rob Meyer for cs73 final project 14F
#Description: This file holds a class that stores data necessary to successfully implement out svd and svm models.
#  It holds information about the sentiment dictionary we are using, the stock data for a date range, the valid dates
# in the date range, and the newspaper data

from collections import defaultdict
from random import shuffle
import stock_downloader
import sentiment
import os
import json

SENTIMENT_FILE = "pitt_lexicon.tff"

class dataStore:
    def __init__(self, trainDirectory, date1, date2):
        self.sentDict = sentiment.SentimentDict()
        self.sentDict.loadSentimentsPitt(SENTIMENT_FILE)

        self.textDict = defaultdict(list)               #store the text from the news articles
        self.scoreDict = {}                             #sentiment scores for articles
        self.stockDict = {}                             #stock data
        self.authorDict = defaultdict(dict)             #optional author dict if want to factor in authors
        self.fillDicts(trainDirectory, date1, date2)
        
        self.dates = list(self.stockDict.keys())        #list of valid dates
        
        
    #helpful get methods    
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
                f = open(os.path.join(root, f), 'r')     #read in the NYT articles for the abstract/lead paragraph
                news = json.loads(f.read())
                if not news:
                    continue
                name = f.name.split('/')[1].strip("#")  #get the name as YYYYMMDD string
                authors = defaultdict(int)
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

                    if item['byline']:
                          
                        if len(item['byline']['person']) != 0:
                            for contributor in range (len(item['byline']['person'])):
                                if item['byline']['person'][contributor].get('lastname'):
                                    authors[item['byline']['person'][contributor]['lastname']] += 1
  
                                  
                        elif item['byline'].get('organization'):
                            authors[item['byline']['organization']] += 1
  
                self.authorDict[name] = authors
                self.scoreDict[name] = (posTotal, negTotal, posTotal + negTotal, superPos, superNeg)  #add a triple as the value for each date
            break
    
    #function that gets the stock data
    def fillStockDict(self, date1, date2):
        databank = stock_downloader.stockDatabank()
        databank.download()
        
        #need to make sure that only have the same dates in each dict
        removeList = []
        for date in self.textDict:
            if databank.dateList.has_key(date) and date > date1 and date < date2:
                self.stockDict[date] = (databank.getStockDirections(date), databank.getStockReturns(date)[0],databank.getStockReturns(date)[1], databank.getStockReturns(date)[2])
            else:
                removeList.append(date)
        for date in removeList:
            self.textDict.pop(date)
            self.scoreDict.pop(date)
            