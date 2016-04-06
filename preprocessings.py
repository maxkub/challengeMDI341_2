from scipy import io
import pandas as pd
import numpy as np

def train_to_coordinates(num):

    train = io.loadmat('/home/max/projects/challengeMDI343/data/data_train.mat')
    
    # dict Id:Label of the gallery
    gallery = {train['galleryId'][i][0]: train['galleryLabel'][i][0] for i in range(len(train['galleryLabel']))}
    
    resultsId = train['resultsId']
    resultsScore = train['resultsScore']
    probeId = train['probeId']
    probeLabel = train['probeLabel']

    zeros = [0 for i in range(12)]
    coords = []

    for probe in range(len(resultsId)):
        for i in range(resultsId.shape[1]):
            for j in range(num):
                
                c = list(zeros)

                # label of the data point
                c[8] = resultsId[probe,i,j]
                c[9] = gallery[resultsId[probe,i,j]]

                # label of the probe
                c[10] = probeId[probe][0]
                c[11] = probeLabel[probe][0]

                # coordinate of the data point
                c[i] = resultsScore[probe,i,j]

                coords.append(c)

    coords = np.asarray(coords)
    
    # name of columns for the dataframe
    cols = ['x'+str(i+1) for i in range(8)] + ['point_id','point_label', 'probe_id', 'probe_label']
    
    df = pd.DataFrame(data = coords, columns=cols)
    
    df['sim'] = df.apply(lambda row: similarity(row['point_label'], row['probe_label']), axis=1)
    
    return df
    
    
def test_to_coordinates(num):
    
    test = io.loadmat('/home/max/projects/challengeMDI343/data/data_test.mat')
    
    # dict Id:Label of the gallery
    gallery = {test['galleryId'][i][0]: test['galleryLabel'][i][0] for i in range(len(test['galleryLabel']))}

    resultsId = test['resultsId']
    resultsScore = test['resultsScore']
    probeId = test['probeId']

    zeros = [0 for i in range(11)]
    coords = []

    for probe in range(len(resultsId)):
        for i in range(resultsId.shape[1]):
            for j in range(num):
                
                c = list(zeros)

                # label of the data point
                c[8] = resultsId[probe,i,j]
                c[9] = gallery[resultsId[probe,i,j]]

                # Id of the probe
                c[10] = probeId[probe][0]

                # coordinate of the data point
                c[i] = resultsScore[probe,i,j]

                coords.append(c)

    coords = np.asarray(coords)
    
    # name of columns for the dataframe
    cols = ['x'+str(i+1) for i in range(8)] + ['point_id','point_label', 'probe_id']
    
    df = pd.DataFrame(data = coords, columns=cols)
    
    df['sim'] = df.apply(lambda row: similarity(row['point_label'], row['probe_label']), axis=1)
    
    return df

    
def similarity(label1, label2):
    if label1 == label2:
        return 1
    else:
        return 0
    
    
    