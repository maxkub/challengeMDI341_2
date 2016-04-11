from scipy import io
import numpy as np

def train_test_indexes(size, test_size, seed=42):
    """
    Returns randoms train and test indexes, with the proprotion defined by test_size
    """
    tot_index = np.asarray(range(size))
    
    np.random.seed(seed)
    test_index = np.random.choice(tot_index, size=int(size*test_size), replace=False)
    
    train_index = np.setdiff1d(tot_index, test_index)
    
    return train_index, test_index
        