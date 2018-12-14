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
from itertools import groupby


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


short_neg = open("/Users/Desktop/cpsc/negative.csv" , "rb").read()
short_pos = open("/Users/Desktop/cpsc/positive.csv" , "rb").read()
short_neutral = open("/Users/Desktop/cpsc/neutral.csv" , "rb").read()
short_sarcastic = open("/Users/Desktop/cpsc/sarcasm_lines.txt" , "rb").read()


all_words = []
documents = []

allowed_word_types = ["J"]
 
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    neg = nltk.pos_tag(words)
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            
            all_words.append(w[0].lower())
            
for p in short_neutral.split('\n'):
    documents.append( (p, "neutral") )
    words = word_tokenize(p)
    neutral = nltk.pos_tag(words)
    for w in neutral:
        if w[1][0] in allowed_word_types:
            
            all_words.append(w[0].lower())
            
for p in short_sarcastic.split('\n'):
    documents.append( (p, "sarcas") )
    words = word_tokenize(p)
    sarcastic = nltk.pos_tag(words)
    for w in sarcastic:
        if w[1][0] in allowed_word_types:
            
            all_words.append(w[0].lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

featuresets = [(find_features(rev),category)for(rev,category) in documents]
random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets [:10000]
testing_set = featuresets [10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print(" Original Naive Bayes Algo accuracy percent:",(nltk.classify.accuracy(classifier,testing_set))*100)
#classifier.show_most_informative_features(30)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

voted_classifier = VoteClassifier(classifier)  
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier,testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats) 


