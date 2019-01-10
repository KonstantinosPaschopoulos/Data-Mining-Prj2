import pandas as pd
import numpy as np
from ast import literal_eval
from gmplot import gmplot
from haversine import haversine
from fastdtw import fastdtw
import csv
import operator
import time

def main():
    hole_time=time.time()
    trainSet = pd.read_csv(
    'train_set.csv',
    converters={"Trajectory": literal_eval}
    )
    testSet = pd.read_csv(
    'test_set_a1.csv',
    converters={"Trajectory": literal_eval}
    )
    for i in range(0,5):
        start_time = time.time()
        lon=[]
        lat=[]
        dist=[]
        param=[]
        for y in range(0,len(testSet['Trajectory'][i])):
            temp=[testSet['Trajectory'][i][y][1], testSet['Trajectory'][i][y][2]]
            lon.append(testSet['Trajectory'][i][y][1])
            lat.append(testSet['Trajectory'][i][y][2])
            param.append(temp)
        for k in range(1, len(trainSet)):
            paramtrain=[]
            for y in range(0,len(trainSet['Trajectory'][k])):
                temp=[trainSet['Trajectory'][k][y][1], trainSet['Trajectory'][k][y][2]]
                paramtrain.append(temp)
            distance, path = fastdtw(np.asarray(param),np.asarray(paramtrain), dist=haversine)
            dist.append((distance,trainSet['journeyPatternId'][k],k))
        sortedDist=sorted(dist, key=operator.itemgetter(0))
        elapsed_time = time.time() - start_time
        gmap = gmplot.GoogleMapPlotter( lat[len(lat)/2],lon[len(lon)/2], 11)
        gmap.plot( lat,lon, 'forestgreen', edge_width=9)
        name="test_"+str(i+1)+"_time_"+str(elapsed_time)+".html"
        gmap.draw(name)
        for y in range(0,5):
            l=sortedDist[y][2]
            lontrain=[]
            lattrain=[]
            for m in range(0,len(trainSet['Trajectory'][l])):
                lontrain.append(trainSet['Trajectory'][l][m][1])
                lattrain.append(trainSet['Trajectory'][l][m][2])
            gmap = gmplot.GoogleMapPlotter( lattrain[len(lattrain)/2],lontrain[len(lontrain)/2], 11)
            gmap.plot(lattrain, lontrain, 'forestgreen', edge_width=9)
            name="test_"+str(i+1)+"_neighbor_"+str(y)+"_jp_"+str(sortedDist[y][1])+"dist"+str(sortedDist[y][0])+".html"
            gmap.draw(name)
    hole_time=time.time()-hole_time

if __name__ == "__main__":
    main()
