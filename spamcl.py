# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:02:02 2018

@author: Chitranjan_Chhaba
"""

import os as os
import pandas as pd
import numpy as np
import re
import nltk
from email.parser import Parser
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
dir='/scratch/hemantk/chitranjan/dat1'
x=[]
y=[]
spaml=[]
for dirn,sundir,finam in os.walk(dir):
    print(dirn,'\n')
    print(os.path.split(dirn)[0],' dm45 ',os.path.split(dirn)[1])
    if(os.path.split(dirn)[1]=='spam'):
        for fn in finam:
            with open(os.path.join(dirn,fn),encoding="latin-1") as f:
                data=f.read()
                email2 = Parser().parsestr(data)
                x.append(email2.get_payload())
                y.append(1)
    if(os.path.split(dirn)[1]=='ham'):
        for fn in finam:
            #print('maku')
            with open(os.path.join(dirn,fn),encoding="latin-1") as f:
                 data=f.read()
                 email = Parser().parsestr(data)
                 x.append(email.get_payload())
                 y.append(0)
corpus = []
for i in range(0, len(x)):
    review = re.sub('[^a-zA-Z]', ' ', x[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()
y = np.asarray(y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#accuraty = 92%
#using ann
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 2500, init = 'uniform', activation = 'sigmoid', input_dim = 2500))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 1250, init = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)