import pandas as pd
import numpy as np
import jieba
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random
from collections import Counter

def read_file(review_file):
    """
    Arguments:
    review_file -- file's name 
    
    Returns:
    e.g. ['东西,很好!!!'] -> ['东西','很','好']
    """  
    df = pd.read_table(review_file, sep='delimiter', header = None)
    mask = [False if ((item == "na") or (item == "没有描述")) else True for item in df[0]]
    df = df[mask]
    skip_mark = '\s|[。？！，、；：「」『』（）〔〕【】—　…　–　．《》〈〉“”]|[,./?:;!@#$%^&*()`~ -_=+<>\"\']'
    len_df = len(df[0])
    review = []
    for i in range(len_df):
        if (i % 10000 == 0):
            print('Processing %d reviews.'%(i))
        segments = jieba.cut(df[0].iloc[i])
        review.append([re.sub(skip_mark,'',element) \
                       for element in segments if (re.sub(skip_mark,'',element) != '')])   
    return list(filter(None, review))

def test_train_generator(review_pos, review_neg):
    """
    Arguments:
    pos_review -- positive review list, its element is list of word for each review
    neg_review -- nagative review list, its element is list of word for each review
    
    Returns:
    X_train_set -- shuffled train data set (mixed with positive and negative reviews). e.g. ['东西','很','好']
    y_train -- shuffled label for train data. e.g. [1, 0]
    X_test_set -- shuffled test data set (mixed with positive and negative reviews) 
    y_test -- shuffled label for test data.
    """  
    label_pos = list(np.ones(len(review_pos)))     #generate label 1 for positive review
    label_neg = list(np.zeros(len(review_neg)))    #generate label 0 for negative review
    
    #seperate train, test data with 80%/20% rules
    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(review_pos, label_pos, test_size=0.2, random_state=1)
    X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(review_neg, label_neg, test_size=0.2, random_state=1)
    X_train_set = X_train_pos + X_train_neg
    y_train = y_train_pos + y_train_neg
    X_test_set = X_test_pos + X_test_neg
    y_test = y_test_pos + y_test_neg
    
    #combine and shuffle the positive/negative reviews in train/test set
    combined_train = list(zip(X_train_set, y_train))
    random.shuffle(combined_train)
    X_train_set, y_train = zip(*combined_train)
    
    combined_test = list(zip(X_test_set, y_test))
    random.shuffle(combined_test)
    X_test_set, y_test = zip(*combined_test)

    #encode label with one-hot encoder method
    y_train = OneHotEncoder().fit_transform(np.array(y_train).reshape(-1, 1)).todense()
    y_test = OneHotEncoder().fit_transform(np.array(y_test).reshape(-1, 1)).todense()
    return X_train_set, y_train, X_test_set, y_test

def word_to_index(review_all, drop_percentile):
    """
    Arguments:
    review_all -- review list, its element is list of word for each review
    drop_percentile -- percentage to choose cut-off counts  
    
    Returns:
    word_to_int -- a dictionary containing the each word mapped to its index
    """  
    words = [word for review in review_all for word in review]     #generate word pool
    counts = Counter(words)                                        #count frequncy of each word
    cutoff = np.percentile(list(counts.values()),drop_percentile)  #choose cutoff based on percentage
    counts = {key:value for key, value in counts.items() if value >= cutoff} #filter words with frequenct >= cutoff
    word_to_int = {key: index for index, key in enumerate(counts.keys(), 1)}
    word_to_int['UNK'] = len(word_to_int)+1
    return word_to_int

def sentences_to_indices(X, word_to_index, max_len):
    """
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """    
    m = X.shape[0]                                   # number of training examples
    X_indices = np.zeros((m, max_len))               # Initialize X_indices as a numpy matrix of zeros 
    for i in range(m):                               # loop over training examples
        sentence_words = X[i]        
        j = 0
        for word in sentence_words:                  # Loop over the words of sentence_words
            if (word in word_to_index.keys()):       # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[word]
            else:
                X_indices[i, j] = word_to_index['UNK']
            j = j + 1
    return X_indices
