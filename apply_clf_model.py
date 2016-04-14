from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from scipy import io

def apply_clf_model(df, clf, train=True):
    
    # predict similarity using the trained classifier
    predSim = predict_sim(clf, df.iloc[:,:8], 1e5)    
    df['predsim'] = predSim
    
    groups = df.groupby('probe_id')
    num_cores = multiprocessing.cpu_count()
    # find the label of the probes, based on the predicted similary objects in the gallery
    predLabel = Parallel(n_jobs=num_cores - 1)(delayed(find_label)(probe_id, group) for probe_id, group in groups)
    predLabel = { k.items()[0][0]: k.items()[0][1] for k in predLabel }
    
    print predLabel
    
    if train == True:
        train = io.loadmat('/home/max/projects/challengeMDI343/data/data_train.mat')
        probes_id = train['probeId'][:,0]
    else:    
        test = io.loadmat('/home/max/projects/challengeMDI343/data/data_test.mat')
        probes_id = test['probeId'][:,0]

    preds = np.asarray([int(predLabel[i]) for i in probes_id])
    
    return preds

    



def predict_sim(clf, data, batch_size):
    
    n = int(len(data)/batch_size)+1
    print data.shape, n
    #print data.head()
    
    pred = []
    for i in range(n):
        
        #try:
            print i,'/',n
            print int(batch_size*i), int(batch_size*(i+1))
            #print data.iloc[batch_size*i:batch_size*(i+1),:].head()
            sim = clf.predict(data.iloc[batch_size*i:batch_size*(i+1),:]).tolist()
            #print 'sim', sim
            pred.extend(sim)
            
        #except IndexError:
        #    pass
            
    return pred


def find_label(probe_id, group):
    #print k, 'probe_id:', name
    
    # finding the predicted similar points and their labels
    try:
        sim_points = group[group['predsim']==1]
        sim_points_labels = sim_points['point_label'].values.astype('int')

        uniques, count = np.unique(sim_points_labels, return_counts=True)
        
        index = np.argwhere(count == np.max(count)).ravel()
        #print 'index', uniques, count, index
        
        if len(index) > 1:
            #print 'multiple max'
            raise ValueError
        else:
            label = uniques[np.argmax(count)]
        
    except ValueError:
        cols = ['x'+str(i+1) for i in range(8)]
        labels = []
        for col in cols:
            dat = group[group[col] != 0]
            #print 'dat1', dat
            dat = dat.iloc[:7,:]
            dat = dat['point_label'].tolist()
            labels.append(dat)
            
        labels = np.asarray(labels)
        
        uniques, count = np.unique(labels, return_counts=True)
        #print 'labels', labels
        #print 'uniques', uniques
        #print 'count', count
        
        try:
            label = uniques[np.argmax(count)]
        except ValueError:
            print group.head()
            print 'labels', labels
            print 'uniques', uniques
            print 'count', count
            label = group['point_label'].iloc[0]
        
        #sim_points_labels = group['point_label'].values.astype('int')
        
        #uniques, count = np.unique(sim_points_labels, return_counts=True)
        #label = uniques[np.argmax(count)]
    return {int(probe_id):int(label)}
    