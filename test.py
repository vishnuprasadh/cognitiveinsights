#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 21:20:11 2017

@author: vishnuhari
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

address = pd.read_csv('addresses.csv',names=['address1','address2','address3','isvalid'],header=0);
address.head()
X = address.iloc[:,0:3].values
Y = address.iloc[:,-1].values
words =[]
for index in range(0,len(X)):
    words.append( ' '.join((X[index,0], X[index,1], X[index,2])))

#words.append(''.join((address_1,address_2,address_3)))

cv = CountVectorizer()
X1 = cv.fit_transform(words)

print(cv.get_feature_names())

tfid = TfidfTransformer()
tX = tfid.fit_transform(X1)


print(tfid.get_params())

X_train,X_test, Y_train,Y_test = train_test_split(X1,Y,test_size=0.2,random_state=12)

classifier = MultinomialNB()
X1 = classifier.fit(X_train,Y_train)
Y_predict = classifier.predict(X_test)

cm = confusion_matrix(Y_test,Y_predict)


cv1 = CountVectorizer()
Ynew = cv1.fit_transform(["test 293 test test"])
Ypred = classifier.predict(Ynew)


