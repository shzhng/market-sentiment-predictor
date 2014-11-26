#svm.py
# You can uncomment features to add or remove features and change the bagofwords function in rawdata_to_vectors to decide
# which feats to use (best or all)
from __future__ import division
import numpy.linalg
from collections import defaultdict
from sklearn import svm
import scipy
from dataStore import dataStore
import string


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
                #loop code for randomly going through the days in a random order
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

def rawdata_to_vectors(filename, date1, date2, ndims):
    """reads raw data, maps to feature space,
    returns a matrix of data points and labels"""
    data = dataStore(filename, date1, date2)
    labels = numpy.zeros((len(data.getText()),), dtype = numpy.int)  #gender labels for each user

    dates = data.getDates()
    for date in range(len(dates)):
        if data.getStocks()[dates[date]][0] == 1:
            labels[date] = 1
        else:
            labels[date] = -1

    #DECIDE WHICH bagofwords-like functon to run
#     representations, numfeats = allFeats(data)      #uncomment if you want to play around with all features
    representations, numfeats = bestFeats(data)   # uncomment to run with best feautures we found

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

#     print "Converted to matrix representation"
    return points, labels, data


def bestFeats(data):
    """represents data in terms of counts of different features
    uncomment various portions to add features that will be considered in the svm model
    returns representations of data points as a dictionary, and number of features"""
    features = {}   #mapping of features to indices
    cur_index = -1
    representations = [] #rep. of each data point in terms of feature values
    cur_index += 1
    features["*PERCENT*"] = cur_index
    cur_index += 1
    features["*NEG*"] = cur_index

    for date in data.getDates():
        i = data.getDates().index(date)
        representations.append(defaultdict(float))
        for abstract in data.getText()[date]:
            flag = False
            for word in abstract.split():
                word.strip().lstrip(string.punctuation).rstrip(string.punctuation)
                
                #check if the word is a negator or an  intensifier, if so flag that we need to check the next word in context ( intensifer NEXT )
                if word in ['not', 'very', 'so', 'really', 'extremely', 'quite', 'remarkably', 'somewhat', 'moderately', 'super']:
                    flag = True
                    flaggedWord = word
                    continue

                if not data.getSentiments().has_key(word):  
                    flag = False
                    continue

                if flag:
                    word = flaggedWord  + " "  + word
                    if not data.getSentiments().has_key(word):  
                        flag = False
                        continue
                    if word in features:
                        feat_index = features[word]
                    else:
                        cur_index += 1
                        features[word] = cur_index
                        feat_index = cur_index
                    flag = False
                else:
                    if word in features:
                        feat_index = features[word]
                    else:
                        cur_index += 1
                        features[word] = cur_index
                        feat_index = cur_index
                representations[i][feat_index] += 1

            representations[i][features["*PERCENT*"]] = data.getScores()[date][0]/ (data.getScores()[date][0] - data.getScores()[date][1])
            representations[i][features["*NEG*"]] = data.getScores()[date][1]
    return representations, cur_index+1


def allFeats(data):
    """represents data in terms of counts of different features
    uncomment various portions to add features that will be considered in the svm model
    returns representations of data points as a dictionary, and number of features"""
#     print contents
    #feature_counts = defaultdict(int)  #total count of each feature, so we can ignore 1-count features UNCOMMENT IF NECESSARY
    features = {}   #mapping of features to indices
    cur_index = -1
    representations = [] #rep. of each data point in terms of feature values
    '''COMMENTED OUT, CAN BE USED IF USING N-GRAM MODEL OR GETTING AUTHOR INFORMATION
#     for date in data.getDates():
#         for abstract in data.getText()[date]:
#             for word in abstract.split():
#                 feature_counts[word]+=1
#
#         for author in data.getAuthors()[date]:
#             feature_counts[author]+=1
    '''

#TO ADJUST THESE FEAUTRES, UNHIGHLIGHT THE FEATURE AND THE cur_index LINE ABOVE IT THEN
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
#     cur_index += 1
#     features["*PREV1*"] = cur_index
#     cur_index += 1
#     features["*PREV2*"] = cur_index
#     cur_index += 1
#     features["*PREV3*"] = cur_index
#     cur_index += 1
#     features["*PREV4*"] = cur_index
#     cur_index += 1
#     features["*PREV5*"] = cur_index
#     cur_index += 1
#     features["*PREV6*"] = cur_index

    for date in data.getDates():
        i = data.getDates().index(date)
        representations.append(defaultdict(float))
        '''       
        #  IF WANT TO ADD AUTHORS ADD A FEAUTURE, UNCOMMENT THIS
        #         for author in data.getAuthors()[date]:
        #             if author in features:
        #                 feat_index = features[author]
        #             else:
        #                 cur_index += 1
        #                 features[author] = cur_index
        #                 feat_index = cur_index
        #             representations[i][feat_index] += 1
        #
        # '''
        for abstract in data.getText()[date]:

#             for word in abstract.split():
# #                 word.strip().lstrip(string.punctuation).rstrip(string.punctuation)
#                 if word[-1] != '.':
#                     continue  #no impact'trending', 'benefits', 'depression', 'bull', 'hot', 'improvement'
#                     #negative impact million, trillion, billion, growth, revenue,performance, salary, market
#
#                 if word in features:
#                     feat_index = features[word]
#                 else:
#                     cur_index += 1
#                     features[word] = cur_index
#                     feat_index = cur_index
#                 representations[i][feat_index] += 1




            flag = False
            for word in abstract.split():
                word.strip().lstrip(string.punctuation).rstrip(string.punctuation)
                
                #check if the word is a negator or an  intensifier, if so flag that we need to check the next word in context ( intensifer NEXT )
                if word in ['not', 'very', 'so', 'really', 'extremely', 'quite', 'remarkably', 'somewhat', 'moderately', 'super']:
                    flag = True
                    flaggedWord = word
                    continue

                if not data.getSentiments().has_key(word) and word not in []:  #no impact'trending', 'benefits', 'depression', 'bull', 'hot', 'improvement'
                    #negative impact million, trillion, billion, growth, revenue,performance, salary, market
                    flag = False
                    continue

                if flag:
                    word = flaggedWord  + " "  + word
                    if not data.getSentiments().has_key(word) and word not in []:  #no impact'trending', 'benefits', 'depression', 'bull', 'hot', 'improvement'
                    #negative impact million, trillion, billion, growth, revenue,performance, salary, market
                        flag = False
                        continue
                    if word in features:
                        feat_index = features[word]
                    else:
                        cur_index += 1
                        features[word] = cur_index
                        feat_index = cur_index
                    flag = False
                else:
                    if word in features:
                        feat_index = features[word]
                    else:
                        cur_index += 1
                        features[word] = cur_index
                        feat_index = cur_index
                representations[i][feat_index] += 1
                
            #****TO SELECT FEATURE: UNCOMMENT FEATURE HERE AND COMMENT NOT USED FEAUTURES
            #MAKE SURE THE COMMENTED FEAUTRES MATCH THE UNCOMMENTED FEATURES ABOVE IN THE LARBE CHUNK OF COMMNETED FEATURES   
            representations[i][features["*PERCENT*"]] = data.getScores()[date][0]/ (data.getScores()[date][0] - data.getScores()[date][1])
#            representations[i][features["*POS*"]] = data.getScores()[date][0]
            representations[i][features["*NEG*"]] = data.getScores()[date][1]
#             representations[i][features["*NET*"]] = data.getScores()[date][2]/len(data.getText()[date])
#             representations[i][features["*NUM_ARTICLES*"]] = len(data.getText()[date])
#             representations[i][features["*SUPER_POS*"]] = data.getScores()[date][3]/len(data.getText()[date])
#             representations[i][features["*SUPER_NEG*"]] = data.getScores()[date][4]/len(data.getText()[date])
#             index = data.getDates().index(date)
#             index -= 1
#             newDate = data.getDates()[index]
#             representations[i][features["*PREV1*"]] = data.getScores()[newDate][1]
#             index -= 1
# #             newDate = data.getDates()[index]
# #             representations[i][features["*PREV2*"]] = data.getScores()[newDate][1]
#             index -= 1
#             newDate = data.getDates()[index]
#             representations[i][features["*PREV3*"]] = data.getScores()[newDate][1]
#             index -= 1
#             newDate = data.getDates()[index]
#             representations[i][features["*PREV4*"]] = data.getScores()[newDate][1]
#             index -= 1
#             newDate = data.getDates()[index]
#             representations[i][features["*PREV5*"]] = data.getScores()[newDate][1]
#             index -= 1
#             newDate = data.getDates()[index]
#             representations[i][features["*PREV6*"]] = data.getScores()[newDate][1]

#             if i == 0 or i == 1:
#                 representations[i][features["*PREV_DATE*"]] = 0
#             else:
#                 prev = data.getDates()[data.getDates().index(date)-1]
#                 prev2 = data.getDates()[data.getDates().index(date)-2]
#                 representations[i][features["*PREV_DATE*"]] = data.getStocks()[prev][0] + data.getStocks()[prev2][0]
    return representations, cur_index+1


if __name__=='__main__':
    date1 = "20100101"
    date2 = "20141120"
    points, labels, data = rawdata_to_vectors('data', date1, date2, ndims=None)

    ttsplit = int(numpy.size(labels)/10)  #split into train and test 85-10
    traindates, testdates = numpy.split(numpy.array(data.getDates()), [ttsplit*8.5])
    traindata, testdata = numpy.split(points, [ttsplit*8.5])
    trainlabels, testlabels = numpy.split(labels, [ttsplit*8.5])

    numfeats = numpy.size(traindata, axis=1)
    svc = svm.LinearSVC()

    traindata = scipy.sparse.csr_matrix(traindata, dtype=numpy.float_)
    testdata = scipy.sparse.csr_matrix(testdata, dtype=numpy.float_)

    svc.fit(traindata, trainlabels)
    print 'Done Training'
    print svc.score(testdata, testlabels)*100, "% rate on test data"

