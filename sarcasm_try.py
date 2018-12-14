import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import codecs
import io
import os
import sys
import csv
import codecs
from nltk.corpus import stopwords
import re
 

class VoteClassifier(ClassifierI):
    def __init__ (self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        

short_pos = open("positive.csv" , "rb").read()
short_neg = open("negative.csv" , "rb").read()
short_sarcastic = open("sarcasm_lines.txt" , "rb").read()


 

documents = []
for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )

for r in short_sarcastic.split('\n'):
    if r not in short_pos:
        documents.append( (r, "spos") )
    else:
        documents.append( (r, "sneg") )
    
        


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)
short_sarcastic_words = word_tokenize(short_sarcastic)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

for w in short_sarcastic_words:
    all_words.append(w.lower())

    
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:1000]

 
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev),category)for(rev,category) in documents]
 

#positive example
training_set = featuresets [:500]
testing_set = featuresets [500:]
'''
filename = 'data672.txt'


stopset = set(stopwords.words('english'))
filtered =[]
with codecs.open(filename, 'r') as myfile:
    job_titles = [line.decode('latin-1').strip() for line in  myfile.readlines()]

for i in job_titles:
    tokenized_sents = word_tokenize(i)


for w in tokenized_sents:
    if w not in stopset:
        filtered.append(w)

filtered = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in filtered]

 
featuresets1 = [(find_features(rev),category)for(rev,category) in filtered ]
testing_set = featuresets1 [1:500]
'''


classifier = nltk.NaiveBayesClassifier.train(training_set)


print(" Original Naive Bayes Algo accuracy percent:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(20)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)



#we will check the accuracy of all these classifiers now
#LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)


#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier accuracy percent:",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)



#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#print("NuSVC_classifier accuracy percent:",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, LogisticRegression_classifier, SGDClassifier_classifier, BernoulliNB_classifier, MNB_classifier) 
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier,testing_set))*100)

#print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)


