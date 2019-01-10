import pandas as pd
import numpy as np
from ast import literal_eval
from gmplot import gmplot
from haversine import haversine
from fastdtw import fastdtw
import csv
import operator
import time

def lcs(X, Y):
    #https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence
    m = len(X)
    n = len(Y)
    #An (m+1) times (n+1) matrix
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            test=((X[i-1][2],X[i-1][1]))
            train=((Y[j-1][2],Y[j-1][1]))
            if haversine(test,train)<=0.2:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    i=m
    j=n
    lon=[]
    lat=[]
    if C[-1][-1]==0:
        return 1
    while(1):
        if i==0 or j==0:
            break
        if(C[i-1][j]==C[i][j-1])and(C[i][j]>C[i-1][j]):
            i=i-1
            j=j-1
            if i==0 or j==0:
                break
            lon.append(Y[j-1][1])
            lat.append(Y[j-1][2])
        elif(C[i-1][j]==C[i][j-1])and(C[i][j]==C[i-1][j]):
            i=i-1
        elif(C[i-1][j]>C[i][j-1]):
            i=i-1
        elif(C[i-1][j]<C[i][j-1]):
            j=j-1
    return (C[-1][-1],lat,lon)

def main():
    hole_time=time.time()
    trainSet = pd.read_csv(
    'train_set.csv',
    converters={"Trajectory": literal_eval}
    )
    testSet = pd.read_csv(
    'test_set_a2.csv',
    converters={"Trajectory": literal_eval}
    )
    for i in range(0,5):
        start_time = time.time()
        lon=[]
        lat=[]
        for y in range(0,len(testSet['Trajectory'][i])):
            lon.append(testSet['Trajectory'][i][y][1])
            lat.append(testSet['Trajectory'][i][y][2])
        common=[]
        for k in range(1, len(trainSet)):
            common.append((lcs(testSet['Trajectory'][i],trainSet['Trajectory'][k]),trainSet['journeyPatternId'][k],k))
        sortedDist=sorted(common, key=operator.itemgetter(0), reverse=True)
        sortedDist=sortedDist[0:5]
        elapsed_time = time.time() - start_time
        #print elapsed_time
        gmap = gmplot.GoogleMapPlotter( lat[len(lat)/2],lon[len(lon)/2], 11)
        gmap.plot( lat,lon, 'forestgreen', edge_width=9)
        name="test_trip_" + str(i+1) + "_elapsedtime_" + str(elapsed_time) + ".html"
        gmap.draw(name)
        for y in range(0,5):
            l=sortedDist[y][2]
            lontrain=[]
            lattrain=[]
            for m in range(0,len(trainSet['Trajectory'][l])):
                lontrain.append(trainSet['Trajectory'][l][m][1])
                lattrain.append(trainSet['Trajectory'][l][m][2])
            gmap = gmplot.GoogleMapPlotter( lattrain[len(lattrain)/2],lontrain[len(lontrain)/2], 11)
            gmap.plot(lattrain, lontrain, 'forestgreen', edge_width=20)
            name="test_trip_" + str(i+1) + "neighbor"+str(y+1)+"_jp_id_"+str(sortedDist[y][1])+"_matched_"+str(sortedDist[y][0][0])+".html"
            gmap.draw(name)
            gmap.plot(sortedDist[y][0][1], sortedDist[y][0][2], 'red', edge_width=9)
            gmap.draw(name)


if __name__ == "__main__":
    main()
