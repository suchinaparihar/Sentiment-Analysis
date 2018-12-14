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
import matplotlib



stopset = set(stopwords.words('english'))
filtered=[]

filename = '/Users/Desktop/stacksample/Questions.csv'

#job_titles = [line.strip() for line in filename.readlines()]
with codecs.open(filename, 'r') as myfile:
    job_titles = [line.decode('latin-1').strip() for line in  myfile.readlines()]

for i in job_titles:
    tokenized_sents = word_tokenize(i)


for w in tokenized_sents:
    if w not in stopset:
        filtered.append(w)

filtered = [re.sub(r'[^A-Za-z0-9]+', '', x) for x in filtered]


print filtered


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(lines)
tagged = []
 
def process_content():
    try:

       for a in filtered:
           words12 = nltk.word_tokenize(a)
           tagged = nltk.pos_tag(words12)
           
           chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}"""
           chunkParser = nltk.RegexpParser(chunkGram)
           chunked = chunkParser.parse(tagged)
           chunked.draw

    except Exception as e:
      print(str(e))

process_content()




    
