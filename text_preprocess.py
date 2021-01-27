#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re

import gensim, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from tqdm import tqdm
import numpy as np
import pymorphy2

lemma_exceptions = ['твиттер']

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
#Extending stop words from file
with open('RussianStopWords.txt', mode= 'r' , encoding = 'utf8') as file:
    for row in tqdm(file, desc = 'Extending stop words'):    
        stop_words.append(row[:-1]) 
print('Number of stop words: ', len(stop_words))

# In[5]:



def preprocess(corpus): 
    new_corpus = []
    morph = pymorphy2.MorphAnalyzer()
    for sentence in tqdm(corpus, desc="text preprocessing"):
        #remove http links
        sentence = re.sub(r'\b(\w+:|)\/\/.+?( |\Z)','', sentence)
     
        #remove retwits, user names, Q:, A:, and hashtags
        sentence = re.sub(r"@([A-Za-z0-9_-]+)",'', sentence)        

        sentence = re.sub('RT' ,'', sentence)        
        
        sentence = re.sub('(Q|A):' ,'', sentence)
        
        sentence = re.sub(r"#([А-Яа-яA-Za-z0-9]+)",'', sentence)    
        
          
        
        # find and replace happy emoticons    
        sentence = re.sub(r'(?::|;|=)(?:-)?(?:\){1,4}|\{1,4}|D|P|З|3)' , ' веселыйсмайлик ' , sentence)
        sentence = re.sub(r'\){1,5}', ' веселыйсмайлик ',sentence)        
               
        # find and replace sad emoticons
        sentence = re.sub(r'(?::|;|=)(?:-)?(?:\{1,4}|\({1,4}|D|P)' , ' грустныйсмайлик ' , sentence)
        sentence = re.sub(r'\({1,5}', ' грустныйсмайлик ',sentence)      
        
        # find question and exclamation !!!! ???
        sentence = re.sub(r'\!{2,5}|\?{2,5}', ' вопросвосклицание ' , sentence)     
        
        # find alphabetic symbols and make lowercase
        sentence = ' '.join(re.findall(r'[а-яё]+', sentence.lower()))      
        
        # Remove stop words        
        #sentence = re.sub(',', ' ' , sentence) 
        sentence = [word for word in simple_preprocess(sentence) if word not in stop_words]
        
        # Lemmatize words
        
        sentence = [morph.parse(word)[0].normal_form if word not in lemma_exceptions else word for word in sentence]
        
        # remove stop words after lemmatization
        sentence = [word for word in sentence if word not in stop_words and len(word)>3]
        
        new_corpus.append(' '.join(sentence))
        
    return new_corpus

def remove_rare_words(texts, num_below, n_frequent):
     #removing rare words by teir number
    texts_tokenized = [simple_preprocess(doc) for doc in texts]
    dictionary = corpora.Dictionary(texts_tokenized)
    dictionary.filter_extremes(no_below = num_below, no_above = 1)
    dictionary.filter_n_most_frequent(n_frequent)
    out_bow = [dictionary.doc2bow(doc) for doc in texts_tokenized]
    out_corpus = []
    for doc in tqdm(out_bow, desc = 'Creating out corpus'):
        out_corpus.append([dictionary.get(id) for id,value in doc])
    return out_corpus

def remove_rare_often_word(texts, low_value, high_value):
     #removing frequent and rare words
    texts_tokenized = [simple_preprocess(doc) for doc in texts]
    dictionary = Dictionary(texts_tokenized)
    corpus = [dictionary.doc2bow(doc) for doc in texts_tokenized]
    
    tfidf = TfidfModel(corpus, id2word=dictionary)
    corpus_tfidf = tfidf[corpus]
    
    bad_words = []
    for sent_tfidf in tqdm(corpus_tfidf, desc="selecting bad words"):
        bad_words += [id for id, value in sent_tfidf if (value < low_value) or (value > high_value)]
    
    dictionary.filter_tokens(bad_ids=bad_words)
    
    out_bow = [dictionary.doc2bow(doc) for doc in texts_tokenized]
    
    out_corpus = []
    for doc in tqdm(out_bow, desc = 'Creating out corpus'):
        out_corpus.append([dictionary.get(id) for id,value in doc])
    
    
    dict_tfidf = {dictionary.get(id): value for doc in corpus_tfidf for id, value in doc if (value >= low_value) and (value <= high_value)}
    
    return {'texts': out_corpus, 
            'dict_tfidf': dict_tfidf,
            'dictionary':dictionary}

def process_words_for_LDA(texts):  
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
   # trigram = gensim.models.Phrases(bigram[texts], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
   # trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    """ Form Bigrams, Trigrams"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    #texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    return texts

def text_to_id(corpus, word2id):
    return [[word2id.get(token, 1) for token in text] for text in corpus]
# In[ ]:




