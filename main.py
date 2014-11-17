#!/usr/bin/env python2

"""
Main Program for Final Project
CS 73

Robert Meyer, Shuo Zheng, and not Josh Lang



"""

import stock_downloader
import sentiment
import nyt
import os
import json
from scipy import stats


def main():

	#stockDict = stockDatabank()
	#sentimentDict = sentimentDict()
	#articleDownloader = articleDownloader()


	#stockDict.read("stockdata.csv")
	#sentimentDict.loadSentimentsPitt("pitt_lexicon.tff")

	#stockReturns = stockDict.getStockReturns("2013-10-14", "2014-10-14")

	### Download articles for all business days since 11/14/2013

	### get sentiment score for each business day (amalgamated across all articles)

	#sentimentScores = ## All the sentiment scores in a list !!

	#slope, intercept, r_value, p_value, std_err = stats.linregress(stockReturns, sentimentScores)

	#print "The equation is y = " + `slope` + " * x + " + `intercept` + "."
	#print "The p-value is " + `p_value` + "and r is " + `r_value` + "."

    sent = sentiment.SentimentDict()

    sent.loadSentimentsPitt("pitt_lexicon.tff")

    for root, dirs, files in os.walk('data'):
        for f in files:
            posTotal, negTotal = 0, 0
            f = open(os.path.join(root, f), 'r')
            news = json.loads(f.read())
            if not news:
                continue

            for item in news:
                if item['abstract']:
                    pos, neg = sent.generateScore(item['abstract'])
                    posTotal += pos
                    negTotal += neg

            print posTotal, negTotal, posTotal + negTotal

        break

main()
