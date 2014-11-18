# svd
# Create for HW5 CS73 14F Due November 2
# @author: Sravana Reddy
# Modified by Joshua Lang to implement training, testing, and additional bagofwords function
# Description: Perceptron Class that is used to use perceptron learning as part of a vector space model
# to analyze the gender of different twitter users based on twitter data.
#
''' I think it will only run in python'''


from __future__ import division
import numpy
import numpy.linalg
from collections import defaultdict
from random import shuffle
import stock_downloader
import sentiment
import nyt
import os
import json
import string
from scipy import stats
from sklearn import svm
import scipy
import csv


class trainDataStorage:
    def __init__(self, trainDirectory):
        self.sentDict = sentiment.SentimentDict()
        self.sentDict.loadSentimentsPitt("pitt_lexicon.tff")

        self.textDict = defaultdict(list)
        self.scoreDict = {}
        self.stockDict = {}
        self.fillDicts(trainDirectory)

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

    #create a list of the dates we are looking at in a particular order so it matches the place in the label and data vectors
    def getDates(self):
#         x = [i for i in range(traindata.shape[0])]   #generate a list of the indices
#         shuffle(x)                                   # randomize the order
        return sorted(list(self.stockDict.keys()))
#         return sorted(self.dates)

    #fill all of the dictionaries for training
    def fillDicts(self, directory):
        self.fillTextAndScoreDict(directory)
        self.fillStockDict()

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

    def fillStockDict(self):
        databank = stock_downloader.stockDatabank()
        databank.download()
        removeList = []
        for date in self.textDict:
            if databank.dateList.has_key(date):
                self.stockDict[date] = (databank.getStockDirections(date), databank.getStockReturns(date)[0],databank.getStockReturns(date)[1], databank.getStockReturns(date)[2] )
            else:
                removeList.append(date)
        for date in removeList:
            self.textDict.pop(date)
            self.scoreDict.pop(date)

#Function to reduce the dimensions of the model (written by Sravana Reddy)
def dimensionality_reduce(data, ndims):
    U, s, Vh = numpy.linalg.svd(data)
    sigmatrix = numpy.matrix(numpy.zeros((ndims, ndims)))
    for i in range(ndims):
        sigmatrix[i, i] = s[i]
    return numpy.array(U[:, 0:ndims] * sigmatrix)

class Perceptron:
    def __init__(self, numfeats):
        self.numfeats = numfeats
        self.w = numpy.zeros((numfeats+1,))   #+1 including intercept
        self.w[0] = 1
        self.alpha = 1  # add the learning rate as a variable for the regular training

        #variables for the average perceptron training
        self.alphaAvgPer = 1    # add the learning rate as a variable for the avg. perception training
        self.cachedW = numpy.zeros((numfeats+1,))    #variable that trying to use for the average perceptron

    #function that trains the vector space model for the regular perceptron learning (you must check the call
    # in the main method to make train function is called instead of trainAvgPerceptron)
    def train(self, traindata, trainlabels, max_epochs):
        traindata = numpy.c_[ traindata, numpy.ones(traindata.shape[0]) ]  #add dummy feature column of ones two end
        loops = 0
        #loop through at least as many epochs as are passed as a parameter (or until no mistakes
        while loops < max_epochs:

            mistakes = 0  #initialize the number of mistakes to 0

#             #COMMENT OUT TO READ TWEETS IN ORDER
#             #loop code for randomly going through the tweets in a random order
#             x = [i for i in range(traindata.shape[0])]   #generate a list of the indices
#             shuffle(x)                                   # randomize the order
#             for row in x:

            #COMMENT OUT TO READ THE TWEETS IN RANDOM ORDER
            #loop code to read the tweets in order
            for row in range(traindata.shape[0]):

                value = numpy.dot(traindata[row], self.w)  # get the dot product of the data point and the weight vector
                #increment the mistake counter and update the weight if the dot product and the label's signs do not match
                if value < 0 and trainlabels[row] == 1:
                    self.w = self.w + self.alpha*traindata[row]
                    mistakes += 1
                elif value > 0 and trainlabels[row] == -1:
                    self.w = self.w - self.alpha*traindata[row]
                    mistakes += 1
#             if there are no mistakes, we can stop
            if mistakes == 0:
                break
            print "Loops: " + str(loops) + "  Mistakes: " + str(mistakes)
            loops+= 1 # increment the number of loops since we have completeed one

        return mistakes #return the number of mistakes we have made

    #function to test the testing data for perception learning
    def test(self, testdata, testlabels, testdates, data):
        testdata = numpy.c_[ testdata, numpy.ones(testdata.shape[0]) ]
        mistakes = 0
        dates = data.getDates()
#         csvout = csv.writer(open("graph.csv", "wb"))
#         csvout.writerow(("Correct", "Guess", "Actual", "Value", "Perceptron"))
        for row in range(testdata.shape[0]):
            value = numpy.dot(testdata[row], self.w)

            #increment mistakes if the
#             stockValue = str(data.getStocks()[testdates[row]][1])
            if value < 0 and testlabels[row] == 1:
#                 print "ERROR: Guessed +; Actually  -:   " + stockValue
#                 csvout.writerow(("W", -1, 1, stockValue, value))
                mistakes += 1
            elif value > 0 and testlabels[row] == -1:
#                 print "WRONG: Guessed -; Actually  +:   " + stockValue
#                 csvout.writerow(("W", 1, -1, stockValue, value))
                mistakes += 1
#             else:
#                 if value>0:
# #                     csvout.writerow(("C", 1, 1, stockValue, value))
#                 else:
#                     csvout.writerow(("C", -1, -1, stockValue, value))
#                 print "CORRECT: Guessed right for:    " + stockValue
        return mistakes

    def trainAvgPerceptron(self, traindata, trainlabels, max_epochs):

            traindata = numpy.c_[ traindata, numpy.ones(traindata.shape[0]) ]  #add dummy feature column of ones to end of traindata
            loops = 0
            counter = 1

            #loop through at most the given number of epochs or until mistakes = 0
            while loops < max_epochs:
                mistakes = 0

                #COMMENT OUT TO READ THE TWEETS IN RANDOM ORDER
                #loop code to read the tweets in order (COMMENT OUT TO USE RANDOM TWEET ORDER)
                for row in range(traindata.shape[0]):

                #COMMENT OUT TO READ TWEETS IN ORDER
                #loop code for randomly going through the tweets in a random order
#                 x = [i for i in range(traindata.shape[0])]  #generate a list of the indices
#                 shuffle(x)                                  #randomize the order
#                 for row in x:


                    value = numpy.dot(traindata[row], self.w)   #get the dot product of the weight vector and the data vector

                    #update the weight vector if the dot product and label for the data point vary in sign
                    #also update the saved cached weight vector
                    if value < 0 and trainlabels[row] == 1:
                        self.w = self.w + self.alphaAvgPer*traindata[row]
                        self.cachedW = self.cachedW + self.alphaAvgPer*counter*traindata[row]
                        mistakes += 1
                    elif value > 0 and trainlabels[row] == -1:
                        self.w = self.w - self.alphaAvgPer*traindata[row]
                        self.cachedW = self.cachedW - self.alphaAvgPer*counter*traindata[row]
                        mistakes += 1
                    counter+= 1     #always increment the counter


                #if there are no mistakes, we can stop
                if mistakes == 0:
                    break
                #update the loops and the learning rate
                loops+= 1
                print "Loops: " + str(loops) + "  Mistakes: " + str(mistakes)
                if loops < 25:
                    self.alphaAvgPer -= .01
                elif loops < 100:
                    self.alphaAvgPer -= .005
#                 elif loops < 50:
#                      self.alphaAvgPer -= 0

            self.w = self.w - (self.cachedW/counter)        #update the weight vector
            return mistakes

def rawdata_to_vectors(filename, ndims):
    """reads raw data, maps to feature space,
    returns a matrix of data points and labels"""
    data = trainDataStorage(filename)
    labels = numpy.zeros((len(data.getText()),), dtype = numpy.int)  #gender labels for each user


#     csvout = csv.writer(open("mydata.csv", "wb"))
#     csvout.writerow(("Date", "StockChange", "Positive", "Negative", "Net Change"))
    dates = data.getDates()
    for date in range(len(dates)):
        if data.getStocks()[dates[date]][0] == 1:
            labels[date] = 1
        else:
            labels[date] = -1
#         datum = str(data.getDates()[date])
#         change = str(data.getStocks()[dates[date]][1])
#         pos = str(data.getScores()[dates[date]][0])
#         neg = str(data.getScores()[dates[date]][1])
#         net = str(data.getScores()[dates[date]][2])
#         csvout.writerow((datum, change, pos, neg, net))
#         print "Date: " + str(data.getDates()[date]) + " Change: " + str(data.getStocks()[dates[date]][1]) + " Pos/Neg/Net:" + str(data.getScores()[dates[date]][0]) + "/" + str(data.getScores()[dates[date]][1]) + "/" + str(data.getScores()[dates[date]][2])

    representations, numfeats = words(data)

    print "Featurized data"

    #convert to a matrix representation
    points = numpy.zeros((len(representations), numfeats))

    for i, rep in enumerate(representations):
        for feat in rep:
            points[i, feat] = rep[feat]

        #normalize to unit length
        l2norm = numpy.linalg.norm(points[i, :])
        if l2norm>0:
            points[i, :]/=l2norm

    if ndims:
        points = dimensionality_reduce(points, ndims)

    print "Converted to matrix representation"
    return points, labels, data

def feats(data):
    """represents data in terms of word counts.
    returns representations of data points as a dictionary, and number of features"""
#     print contents
    feature_counts = defaultdict(int)  #total count of each feature, so we can ignore 1-count features
    features = {}   #mapping of features to indices
    cur_index = -1
    representations = [] #rep. of each data point in terms of feature values
    for date in data.getDates():
        for abstract in data.getText()[date]:
            for word in abstract.split():
                feature_counts[word]+=1


#    cur_index += 1
#    features["*PERCENT*"] = cur_index
#     cur_index += 1
#     features["*POS*"] = cur_index
    cur_index += 1
    features["*NEG*"] = cur_index

    for date in data.getDates():
        i = data.getDates().index(date)
        representations.append(defaultdict(float))
        for abstract in data.getText()[date]:
#            representations[i][features["*PERCENT*"]] = data.getScores()[date][0]/ (data.getScores()[date][0] - data.getScores()[date][1])
#             representations[i][features["*POS*"]] = data.getScores()[date][0]
            representations[i][features["*NEG*"]] = data.getScores()[date][1]

    return representations, cur_index+1

def words(data):
    """represents data in terms of word counts.
    returns representations of data points as a dictionary, and number of features"""
#     print contents
    feature_counts = defaultdict(int)  #total count of each feature, so we can ignore 1-count features
    features = {}   #mapping of features to indices
    cur_index = -1
    representations = [] #rep. of each data point in terms of feature values
    for date in data.getDates():
        for abstract in data.getText()[date]:
            for word in abstract.split():
                feature_counts[word]+=1


    cur_index += 1
    features["*PERCENT*"] = cur_index
#     cur_index += 1
#     features["*POS*"] = cur_index
    cur_index += 1
    features["*NEG*"] = cur_index
#     cur_index += 1
#     features["*NET*"] = cur_index
#     cur_index += 1
#     features["*NUM_ARTICLES*"] = cur_index
#     cur_index += 1
#     features["*SUPER_POS*"] = cur_index
#     cur_index += 1
#     features["*SUPER_NEG*"] = cur_index
#     cur_index += 1
#     features["*PREV_DATE*"] = cur_index


    for date in data.getDates():
        i = data.getDates().index(date)
        representations.append(defaultdict(float))
        for abstract in data.getText()[date]:
            for word in abstract.split():
#                 word.strip().lstrip(string.punctuation).rstrip(string.punctuation)
                if not data.getSentiments().has_key(word) and word not in []:  #no impact'trending', 'benefits', 'depression', 'bull', 'hot', 'improvement'
                    #negative impact million, trillion, billion, growth, revenue,performance, salary, market
                    continue
                if word in features:
                    feat_index = features[word]
                else:
                    cur_index += 1
                    features[word] = cur_index
                    feat_index = cur_index
                representations[i][feat_index] += 1
            representations[i][features["*PERCENT*"]] = data.getScores()[date][0]/ (data.getScores()[date][0] - data.getScores()[date][1])
#             representations[i][features["*POS*"]] = data.getScores()[date][0]
            representations[i][features["*NEG*"]] = data.getScores()[date][1]
#             representations[i][features["*NET*"]] = data.getScores()[date][2]/len(data.getText()[date])
#             representations[i][features["*NUM_ARTICLES*"]] = len(data.getText()[date])
#             representations[i][features["*SUPER_POS*"]] = data.getScores()[date][3]/len(data.getText()[date])
#             representations[i][features["*SUPER_NEG*"]] = data.getScores()[date][4]/len(data.getText()[date])
#             if i == 0 or i == 1:
#                 representations[i][features["*PREV_DATE*"]] = 0
#             else:
#                 prev = data.getDates()[data.getDates().index(date)-1]
#                 prev2 = data.getDates()[data.getDates().index(date)-2]
#                 representations[i][features["*PREV_DATE*"]] = data.getStocks()[prev][0] + data.getStocks()[prev2][0]
    return representations, cur_index+1


if __name__=='__main__':
#     traindata, trainlabels = rawdata_to_vectors('data', ndims=None)#
    points, labels, data = rawdata_to_vectors('newTest', ndims=None)#
#     #added text
#     dateList = data.getDates()
#     print dateList
#     dateArray = numpy.array(data.getDates())
#     print dateArray
    ttsplit = int(numpy.size(labels)/10)  #split into train, dev, and test 80-10-10

    traindates, devdates, testdates = numpy.split(numpy.array(data.getDates()), [ttsplit*8, ttsplit*9])
    traindata, devdata, testdata = numpy.split(points, [ttsplit*8, ttsplit*9])
    trainlabels, devlabels, testlabels = numpy.split(labels, [ttsplit*8, ttsplit*9])

    numfeats = numpy.size(traindata, axis=1)
    svc = svm.SVC(numfeats)
    traindata = scipy.sparse.csr_matrix(traindata, dtype=numpy.float_)
    testdata = scipy.sparse.csr_matrix(testdata, dtype=numpy.float_)
    devdata = scipy.sparse.csr_matrix(devdata, dtype=numpy.float_)
    svc.fit(traindata, trainlabels)

    print str(traindates.shape), "****", str(traindata.shape), "train"
    print str(testdates.shape), "****", str(testdata.shape), "test"
    print str(devdates.shape), "****", str(devdata.shape), "dev"

    print "Training..."

    print svc.score(devdata, devlabels), "% rate on development data"
    print svc.score(testdata, testlabels), "% rate on test data"
