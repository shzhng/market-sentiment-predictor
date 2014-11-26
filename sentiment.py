#!/usr/bin/env python2
#File that reads in the sentiment corpus that we have
#Assumes that the format of the file is that a word is on each line


import string

negaters = {
        'not': True
}

augmenters = {
        'very': True,
        'so': True,
        'really': True,
        'extremely': True,
        'remarkably': True,
        'super': True
}

softeners = {
        'quite': True,
        'somewhat': True,
        'moderately': True
}

class SentimentDict:

    def __init__(self):
        self.sentiments = None

    def loadSentimentsPitt(self, path):
        newDict = {}
        for line in open(path):
            attrs = dict(attr.partition('=')[::2] for attr in line.split())
            word = attrs['word1']
            if attrs['priorpolarity'] == 'positive':
                newDict[word] = 1 if attrs['type'] == 'weaksubj' else 2
            elif attrs['priorpolarity'] == 'negative':
                newDict[word] = -1 if attrs['type'] == 'weaksubj' else -2

        self.sentiments = newDict

    #function that generates a score for a given string of text
    #assuming that value is not always just 1 for positive and -1 for negative since we could weight words
    def generateScore(self, textString):
        pos = 0
        neg = 0
        prevWord = ""
        for word in textString.split():
            word = word.strip().rstrip(string.punctuation).lstrip(string.punctuation) #remove punctuation that might be on the word
            if self.sentiments.has_key(word):
                modifier = 1
                modifier = -1 if prevWord in negaters else modifier
                modifier = 2 if prevWord in augmenters else modifier
                modifier = 0.5 if prevWord in softeners else modifier
                value = self.sentiments[word] * modifier
                if value > 0:
                    pos += value
                elif value < 0:
                    neg += value
        #return a tupple of the value of the positive words, value of the negative values and net score
        return pos, neg

#test that it works on a string
def test():
    test = SentimentDict()

    test.loadSentimentsPitt("pitt_lexicon.tff")

    string = "he aggressively went$ to happy the 'Exciting. rage joyous/ ferocious"
    triple = test.generateScore(string)

    print "Test String: " + string
    print "Pos Total: " + str(triple[0])
    print "Neg Total: " + str(triple[1])
    print "Net Score: " + str(triple[2])

#test()
