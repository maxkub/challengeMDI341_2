from scipy import io
import numpy as np

def train_test_indexes(test_size, seed=42, path='/home/max/projects/challengeMDI343/data/data_train.mat'):
    """
    Returns randoms train and test indexes, with the proprotion defined by test_size
    """
    train = io.loadmat(path)
    
    num_probes = len(train['probeId'].ravel())
    tot_index = np.asarray(range(num_probes))
    
    np.random.seed(seed)
    test_index = np.random.choice(tot_index, size=int(num_probes*test_size), replace=False)
    
    train_index = np.setdiff1d(tot_index, test_index)
    
    return train_index, test_index
        