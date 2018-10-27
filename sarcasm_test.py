import sarcasm_finder as s
import os
import io
import codecs
import sys
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
 
 

stopset = set(stopwords.words('english'))
filtered=[]

filename = 'data672.txt'

#job_titles = [line.strip() for line in filename.readlines()]
with codecs.open(filename, 'r') as myfile:
    job_titles = [line.decode('latin-1').strip() for line in  myfile.readlines()]

for i in job_titles:
    tokenized_sents = word_tokenize(i)


for w in tokenized_sents:
    if w not in stopset:
        filtered.append(w)

filtered = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in filtered]

for a in filtered:
  print (s.sentiment(a))
     
