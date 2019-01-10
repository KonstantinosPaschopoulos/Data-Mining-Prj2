import pandas as pd
import numpy as np
from ast import literal_eval
from haversine import haversine
from fastdtw import fastdtw
import csv
import operator
import time
from sklearn.model_selection import StratifiedKFold
import knn_functions as knn
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def main():
    trainSet = pd.read_csv(
    'train_set.csv',
    converters={"Trajectory": literal_eval}
    )
    testSet = pd.read_csv(
    'test_set_a2.csv',
    converters={"Trajectory": literal_eval}
    )

    paramtrain=[]
    X_train=[]
    for i in range(0,20):   #should be len(trainSet['Trajectory'])
        for y in range(0,len(trainSet['Trajectory'][i])):
            temp=[trainSet['Trajectory'][i][y][1], trainSet['Trajectory'][i][y][2]]
            paramtrain.append(temp)
        X_train.append(paramtrain)

    set(trainSet['journeyPatternId'])		#check categories
    le=preprocessing.LabelEncoder()	#set labels
    le.fit(trainSet['journeyPatternId'][0:20])	#fit them to the number of our categories
    y_train=le.transform(trainSet['journeyPatternId'][0:20])	#transform categories
    set(y_train)

    paramtest=[]
    test=[]
    for i in range(0,1):
        for y in range(0,len(testSet['Trajectory'][i])):
            temp=[testSet['Trajectory'][i][y][1], testSet['Trajectory'][i][y][2]]
            paramtest.append(temp)
        test.append(paramtest)

    k_fold=StratifiedKFold(n_splits=3)  #use only 3 because of the memory problems

    clf=knn.myKNN(5)			# K=5,check knn_functions.py(imported)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(test)

    KNNaccs=cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')
    knn_acc=KNNaccs.mean()
    print "accuracy:" ,knn_acc

    output='testSet_JourneyPatternIDs.csv'
    predicted=le.inverse_transform(y_pred)
    nums=[]
    for i in range(1,len(predicted)+1):
        nums.append(i)
    testingfile=pd.DataFrame({'Test_Trip_ID': nums, 'Predicted_JourneyPatternID': list(predicted)}, columns=['Test_Trip_ID','Predicted_JourneyPatternID'])
    testingfile.to_csv(output,encoding='utf-8',index=False,sep='\t')

if __name__ == "__main__":
    main()
