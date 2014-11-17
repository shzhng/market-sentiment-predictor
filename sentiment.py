#!/usr/bin/env python2
#File that reads in the sentiment corpus that we have
#Assumes that the format of the file is that a word is on each line

positive = "./positive-words.txt"
negative = "./negative-words.txt"

import string

class sentimentDict:

    def __init__(self):
        self.sentiments = None

    def loadSentiments(self):
        newDict = {}
        #load each file
        for line in open(positive):
            newDict[line.strip()] = 1
        for line in open(negative):
            newDict[line.strip()] = -1

        self.sentiments = newDict
        #return newDict

    def loadSentimentsPitt(self, path):

        newDict = {}
        word = ""
        sent = ""

        for line in open(path):
            pieces = line.split()
            for piece in pieces:
                pieceparts = piece.split('=')
                if pieceparts[0] == "word1":
                    word = pieceparts[1]
                if pieceparts[0] == "priorpolarity":
                    sent = pieceparts[1]
            if sent == "positive":
                newDict[word] = 1
            else:
                newDict[word] = -1

        self.sentiments = newDict

    #function that generates a score for a given string of text
    #assuming that value is not always just 1 for positive and -1 for negative since we could weight words
    def generateScore(self, textString):
        pos = 0
        neg = 0
        for word in textString.split():
            word = word.strip().rstrip(string.punctuation).lstrip(string.punctuation) #remove punctuation that might be on the word
            if self.sentiments.has_key(word):
                value = self.sentiments[word]
                if value > 0:
                    pos += value
                elif value < 0:
                    neg += value
        #return a tupple of the value of the positive words, value of the negative values and net score
        return (pos, neg, pos + neg)


#test that it works on a string
def test():
    test = sentimentDict()

    test.loadSentimentsPitt("pitt_lexicon.tff")

    string = "he aggressively went$ to happy the 'Exciting. rage joyous/ ferocious"
    triple = test.generateScore(string)

    print "Test String: " + string
    print "Pos Total: " + str(triple[0])
    print "Neg Total: " + str(triple[1])
    print "Net Score: " + str(triple[2])

test()
