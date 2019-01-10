import pandas as pd
import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import re
import string
from haversine import haversine
from fastdtw import fastdtw
from sklearn.metrics.pairwise import euclidean_distances

def voting(k,testing,KNN_train,y_train):
    pred=[]
    #for each component
    for i in range(len(testing)):
        distances=[]
        for j in range(len(KNN_train)):
            dist, path = fastdtw(np.array(testing[i]),np.array(KNN_train[j]), dist=haversine)
            distances.append((y_train[j],dist))
        sortedDistances=sorted(distances,key=operator.itemgetter(1))
        neighbors=sortedDistances[0:k]
        votes={}
        for y in range(k):
            answer=neighbors[y][0]
            if answer in votes:
                votes[answer]+=1
            else:
                votes[answer]=1
        sortedVotes=sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
        pred.append(sortedVotes[0][0])
    return pred

class myKNN(BaseEstimator, ClassifierMixin):
    def __init__(self,k):
        self.k=k		#number of neighbors

    def fit(self,X,y):
        # Check that X and y have correct shape
        #X, y =check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_=unique_labels(y)
        self.X_=X
        self.y_=y
        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit had been called
        #check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X=check_array(X)
        result=voting(self.k, X, self.X_, self.y_)
        return result
