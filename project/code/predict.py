from random import triangular
import re
from nltk import data
from nltk.util import pr
import numpy as np
import json
import nltk
#from nltk.stem.porter import PorterStemmer  # porter classifier
#from nltk.stem.lancaster import LancasterStemmer    # lancaster classifier
#from nltk.stem import SnowballStemmer
#from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes import NaiveBayesClassifier
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB  # use provided NB for comparison
from sklearn.svm import SVC # use provided SVM for comparison
from googletrans import Translator
import csv
import time

test_num = 400

def Pretreat_news(content):
    word_net_obj=WordNetLemmatizer()
    stop_word_list=stopwords.words('english')
    
    # words division
    words_list=nltk.word_tokenize(content)
    
    # mark the part of speech
    pos_tag_list=nltk.pos_tag(words_list)
    
    # word type normalization
    normalize_words_list=[]
    for word_tag in pos_tag_list:
        if word_tag[1]=='JJ':
            normalize_words_list.append(word_net_obj.lemmatize(word_tag[0], 'a'))
        elif (word_tag[1]=='VB' or word_tag[1]=='VBD' or word_tag[1]=='VBG' or word_tag[1]=='VBN'):
            normalize_words_list.append(word_net_obj.lemmatize(word_tag[0], 'v'))
        elif (word_tag[1]=='NN' or word_tag[1]=='NNS' or word_tag[1]=='NNP' or word_tag[1]=='VBN'):
            normalize_words_list.append(word_net_obj.lemmatize(word_tag[0], 'n'))
        elif word_tag[1]=='RB':
            normalize_words_list.append(word_net_obj.lemmatize(word_tag[0], 'r'))
        else:
            normalize_words_list.append(word_net_obj.lemmatize(word_tag[0]))
            
    # filter stop words
    remind_word_list=[]
    for word in normalize_words_list:
        if word not in stop_word_list:
            remind_word_list.append(word)
            
    pretreated_text=''
    for word in remind_word_list:
        pretreated_text=pretreated_text+word
        pretreated_text=pretreated_text+' '
    
    return pretreated_text

def news_predict(train_sample, train_label, test_sample):
    '''
    training model and predict then return the result
    :param train_sample: news context in original training set <ndarray>
    :param train_label: the corresponding labels in training set <ndarray>
    :param test_sample: news context in original testing set <ndarray>
    :return predict result <ndarray>
    '''
    # instantiate the vectorizer
    vec = CountVectorizer()
    # vectorize the news in training set
    X_train_count_vectorizer = vec.fit_transform(train_sample).toarray()
    # vectorize the news in test set
    X_test_count_vectorizer = vec.transform(test_sample).toarray()

    # instantiate the tf-idf object
    #tfidf = TfidfTransformer()
    # transfer the frequency vectors in training set by tf-idf
    #X_train = tfidf.fit_transform(X_train_count_vectorizer)
    # transfer the frequency vectors in test set by tf-idf
    #X_test = tfidf.transform(X_test_count_vectorizer)
    
    all_data = np.vstack((X_train_count_vectorizer, X_test_count_vectorizer))
    
    pca = PCA(n_components = 100)
    principalComponents = pca.fit_transform(all_data)
    
    X_train_count_vectorizer = principalComponents[:-test_num]
    X_test_count_vectorizer = principalComponents[-test_num:]
    
    t0 = time.perf_counter()
    clf = NaiveBayesClassifier()
    #clf = BernoulliNB(alpha = 0.03)
    #clf = SVC()
    clf.fit(X_train_count_vectorizer, train_label, 2)   # the 3rd arg is k for Laplacian Smoothing
    #clf.fit(X_train_count_vectorizer, train_label)
    result = clf.predict(X_test_count_vectorizer)
    t1 = time.perf_counter()
    print('time: ', 1000*(t1-t0), 'ms')    # time counter
    return result
    
def data_process():
    
    with open("train_translate.json", 'r', encoding="UTF-8") as f:
        all_data = json.load(f)

    # depart test and train
    train_data = all_data[:-test_num]
    test_data = all_data[-test_num:]

    pretreated_train_sample=[]
    i=1
    print(len(train_data))
    for data in train_data:
        pretreated_train_sample.append(Pretreat_news(data['content']))
        print('\nfinish train data ')
        print(i)
        i=i+1
        
    train_label=[]
    for data in train_data:
        train_label.append(str(data['label']))
        
    pretreated_test_sample=[]
    i=1
    for data in test_data:
        pretreated_test_sample.append(Pretreat_news(data['content']))
        print('\nfinish test data ')
        print(i)
        i=i+1
        
    test_label=[]
    for data in test_data:
        test_label.append(str(data['label']))
        
    with open('pretreated_train_sample.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(pretreated_train_sample)
        
    with open('train_label.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(train_label)
        
    with open('pretreated_test_sample.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(pretreated_test_sample)
        
    with open('test_label.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(test_label)
        
def run():
    pretreated_train_sample = []
    train_label = []
    pretreated_test_sample = []
    with open('pretreated_train_sample.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            pretreated_train_sample.append("".join(row))
        
    with open('train_label.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            train_label.append(eval(row[0]))
        
    with open('pretreated_test_sample.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            pretreated_test_sample.append("".join(row))
            
    pretreated_train_sample = np.array(pretreated_train_sample)
    train_label = np.array(train_label)
    pretreated_test_sample = np.array(pretreated_test_sample)
    result = news_predict(pretreated_train_sample, train_label, pretreated_test_sample)
    #print(result);
    #for re in result:
        #print(re, end='')
        #print('\n', end='')
    return result

def f1_score(result):
    test_label = []
    with open('test_label.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            test_label.append(eval(row[0]))
    TP = 0
    FP = 0
    FN = 0
    for i in range(test_num):
        if test_label[i] == 1:
            if result[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[i] == 1:
                FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / (1 / precision + 1 / recall)
    print("TP = {}, FP = {}, FN = {}\nF1 score: {}".format(TP, FP, FN, f1))
        
if __name__ == "__main__":
    # data_process()
    result = run()
    f1_score(result)
    