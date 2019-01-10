import pandas as pd
from ast import literal_eval
from gmplot import gmplot

def main():
    trainSet = pd.read_csv(
    'train_set.csv',
    converters={"Trajectory": literal_eval},
    index_col='tripId', nrows=10
    )
    for i in range(1,7):
        lats=[]
        lons=[]
        name='map_'
        if i==3 :
            continue
        for y in range(0,len(trainSet['Trajectory'][i])):
            lats.append(trainSet['Trajectory'][i][y][2])
            lons.append(trainSet['Trajectory'][i][y][1])
        gmap = gmplot.GoogleMapPlotter(lats[y/2], lons[y/2], 11)
        gmap.plot(lats, lons, 'forestgreen', edge_width=9)
        name+=str(trainSet['journeyPatternId'][i])
        name+='.html'
        gmap.draw(name)

if __name__ == "__main__":
    main()
