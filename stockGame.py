#Class for the interactive game
import numpy
import math

INIT_BALANCE = 10000000
START_SHARES = 100

class stockGame:
    def __init__(self, testdata, testlabels, testdates, data, weightVector):
        #date for the svd model
        self.testdata = numpy.c_[testdata, numpy.ones(testdata.shape[0])]
        self.testlabels = testlabels
        self.testdates = testdates
        self.allDates = data.getDates()
        self.data = data
        self.w = weightVector

        #data for your statistics
        self.yourBalance = INIT_BALANCE
        self.yourShares = START_SHARES
        self.yourTotal = INIT_BALANCE + START_SHARES* float(self.data.getStocks()[self.testdates[0]][2])

        #data for computer
        self.compBalance = INIT_BALANCE
        self.compShares = START_SHARES
        self.compTotal = INIT_BALANCE + START_SHARES* float(self.data.getStocks()[self.testdates[0]][2])

        #date for the starting conditions
        self.startBalance = INIT_BALANCE
        self.startShares = START_SHARES
        self.startTotal = INIT_BALANCE + START_SHARES* float(self.data.getStocks()[self.testdates[0]][2])


    def playGame (self):
        for row in range(self.testdata.shape[0]):
            close = self.getPrediction(row)
            if close:
                self.quit(float(self.data.getStocks()[self.testdates[row]][3]))
                return
        self.quit(float(self.data.getStocks()[self.testdates[self.testdata.shape[0]-1]][3]))

    def getPrediction(self, row):
        openPrice = float(self.data.getStocks()[self.testdates[row]][2])
        closePrice = float(self.data.getStocks()[self.testdates[row]][3])

        value = numpy.dot(self.testdata[row], self.w)
        self.predictionPrompt(value, openPrice)

        delta = self.getUserInput()
        if delta == 'q':
            return True
        self.transaction(int(delta), value, self.testlabels[row], openPrice)

        if self.testlabels[row] == 1:
            print "The stock price increased to " + str(closePrice) + "."
        else:
            print "The stock price decreased to " + str(closePrice) + "."
        return False

    def transaction(self, shares, value, label, openPrice):
        if value < 0:
            self.compBalance += math.fabs(5)*openPrice
            self.compShares -= math.fabs(5)
        if value > 0:
            self.compBalance -= math.fabs(5) * openPrice
            self.compShares += math.fabs(5)
        self.yourBalance -= shares*openPrice
        self.yourShares += shares

    def getUserInput(self):
        try:
            input = raw_input("How many shares would you like to buy/sell today? ")
            if input == 'q':
                return input
            input = int(input)
            return input
        except Exception:
            return self.getUserInput()

    def predictionPrompt(self, value, open):
        print ""
        if value < 0:
            print "Our predictor anticipates the market to go down today."
        elif value > 0:
            print "Our predictor anticipates the market to go up today."
        else:
            print "Our predictor anticipates the market to do nothing today."
        print "Your Balance:" + str(self.yourBalance) + "  Your Shares: " + str(self.yourShares)
        print "Opening Price: " + str(open) + "."

    def quit(self, closingPrice):
        print "*******************"
        print "Thanks for playing."
        print "You have " + str(self.yourShares)  + " shares at " + str(closingPrice)
        print "Your account has "  + str(self.yourBalance) + "."
        total = self.yourBalance + float(self.yourShares* float(closingPrice))
        diff = total- self.yourTotal
        print "Your profit is " + str(diff) + "."

        print "*******************"
        print "The predictor has " + str(self.compShares)  + " shares at " + str(closingPrice)
        print "Your account has "  + str(self.compBalance) + "."
        total = self.compBalance + float(self.compShares* float(closingPrice))
        diff = total - self.compTotal
        print "Your total is "  + str(diff) + "."

        print "*******************"
        print "If you just kept your money in your market, you would have " + str(self.startShares)  + " shares at " + str(closingPrice)
        total = self.startBalance + float(self.startShares* float(closingPrice))
        diff = total - self.startTotal
        print "Your total is "  + str(diff) + "."
