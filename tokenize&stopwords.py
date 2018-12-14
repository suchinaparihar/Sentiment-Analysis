import nltk
import csv
import pandas as pd
import os
import io
import codecs
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


stopset = set(stopwords.words('english'))
filtered=[]

#path where the txt file is stored on the desktop
filename = '/Users/Desktop/data672.txt'

#job_titles = [line.strip() for line in filename.readlines()]
with codecs.open(filename, 'r') as myfile:
    job_titles = [line.decode('latin-1').strip() for line in  myfile.readlines()]

#tokenization
for i in job_titles:
    tokenized_sents = word_tokenize(i)

#removing stopwords
for w in tokenized_sents:
    if w not in stopset:
       filtered.append(w)

filtered = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in filtered]

 
print filtered
