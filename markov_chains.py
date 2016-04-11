import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import itertools

def find_label(ranks, probe_id, eps=1e-2, tol=1e-3, max_rounds=500, select_model=7):
    """
    Function that return the most probable label based on a Markov chain algorithm
    
    INPUTS
    
    ranks        : matrix of ranked items to use as features of the model [n_items, n_ranking_function]
    eps          : value of the epsilon added to the transition matrixes
    tol          : stop criterion to the computation of the equilibrium states
    max_rounds   : the maximum number of iteration in the computation of the equilibrium states
    select_model : selecting the model of markov chain to use (the best model being build_transition8)
    """
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)
    
    # list of markov models
    builds = [build_transition1, build_transition2, build_transition3, build_transition4, build_transition5, build_transition6,
             build_transition7, build_transition8]
    
    model = builds[select_model]
    
    # build transition matrix
    transit = model(ranks, eps)
    
    # get probabilities of each final state of the markov chain
    label_probas = get_equilibrium(transit,uniques, tol, max_rounds)

    # finding the most probable final state
    max_label = uniques[np.argmax(label_probas)]

    return {probe_id: max_label}


def get_equilibrium(transit, uniques, tol, max_rounds):
    """
    Apply recursively the transition matrix to the vector of prababilities, until convergence.
    """
    # initialisation of the probability of each label (or states)
    label_proba2 = np.ones(len(uniques))*1./len(uniques)
    
    # applying the transition matrix
    i = 0
    diff = 1.
    while diff >= tol and i <= max_rounds:
        label_proba1 = label_proba2
        label_proba2 = transit.dot(label_proba1)
        diff = np.linalg.norm(label_proba2 - label_proba1)
        i += 1
        
    return label_proba2

def random_rankings(ranks, n_estimators, max_features=3, eps_min=1e-2, eps_max=0.8, tol=1e-5, max_rounds=100, seed=42, n_jobs=-1):
    """
    Combining multiple rankings based on markov chains, with different transition matrix
    
    INPUTS:
    
    ranks        : matrix of ranked items to use as features of the model [n_items, n_ranking_function]
    n_estimators : number of weak markov chains to compute
    max_features : maximum number of features that can be randomly chosen (<=n_ranking_function)
    eps_min      : minimum value of the epsilon added to the transition matrixes
    eps_max      : maximum value of the epsilon added to the transition matrixes
    tol          : stop criterion to the computation of the equilibrium states
    max_rounds   : the maximum number of iteration in the computation of the equilibrium states
    seed         : initialization of the numpy.random seed
    n_jobs       : number of processor to use in parallel computing, if None: no parallel
    """
    np.random.seed(seed)
    
    # initialisation of the new ranking matrix
    rankings = np.empty([ranks.shape[1], n_estimators])
    
    if n_jobs != None:
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # computing n_estimators weak markov chains in parallel, with the function new_ranking()
        results = Parallel(n_jobs=n_jobs)(delayed(new_ranking)(ranks, max_features, eps_min, eps_max, tol, max_rounds)
                                           for i in range(n_estimators))
        # aggregating the results in np.array
        for i in range(n_estimators):
            rankings[:,i] = results[i] 
        
    else:
        for i in range(n_estimators):
            rankings[:,i] = new_ranking(ranks, max_features, eps_min, eps_max, tol, max_rounds)
        
    return rankings


    
def new_ranking(ranks, max_features, eps_min, eps_max, tol, max_rounds):
    """
    Function used in random_rankings
    """
    
    index = np.asarray(range(ranks.shape[0]))
    
    # list of markov models
    builds = [build_transition1, build_transition2, build_transition3, build_transition4, build_transition5, build_transition6,
             build_transition7, build_transition8]
    
    # choose randomly the number of features to take
    n_features = np.random.randint(2,max_features+1)
        
    # select randomly the n_features to use
    rk = ranks[np.random.choice(index, size=n_features, replace=False),:]
    uniques = np.unique(rk)
        
    # select randomly the Morkov model in the list 'builds'
    #s = np.random.randint(len(builds))
    model = builds[7]
    
    #print s, n_features , '|||'
    
        
    # build the corresponding transition matrix
    transit = model(rk, eps=np.random.uniform(eps_min, eps_max))
        
    # get the equilibrium state of the markov chain
    res = get_equilibrium(transit, uniques, tol, max_rounds)
        
    # sort individual states according to the equilibrium state
    ind = np.argsort(-res)
    
    return uniques[ind][:ranks.shape[1]]



def build_transition1(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
        
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += np.sign(posi-posj)
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.
    
    transit[transit>0.] = 1.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    for i in range(len(transit)):
        probs = transit[:,i]
        count = len(probs[probs!=0.])
        probs[probs == 0] = eps/float(count)
        transit[:,i] = probs

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit




def build_transition2(ranks, eps=5e-1):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
        
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += np.sign(posi-posj)
            
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    for i in range(len(transit)):
        probs = transit[:,i]
        count = len(probs[probs!=0.])
        probs[probs == 0] = eps/float(count)
        transit[:,i] = probs

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit



def build_transition3(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
        
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += np.sign(posi-posj)
    
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    transit = transit + eps

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit



def build_transition4(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
        
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += np.sign(posi-posj)
    
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.
    
    transit[transit>0.] = 1.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    transit = transit + eps

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit




def build_transition5(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
  
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += posi-posj
    
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.
    
    transit[transit>0.] = 1.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    #transit = transit + eps
    for i in range(len(transit)):
        probs = transit[:,i]
        count = len(probs[probs!=0.])
        probs[probs == 0] = eps/float(count)
        transit[:,i] = probs

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit




def build_transition6(ranks, eps=5e-1):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += posi-posj
    
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    for i in range(len(transit)):
        probs = transit[:,i]
        count = len(probs[probs!=0.])
        probs[probs == 0] = eps/float(count)
        transit[:,i] = probs

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit



def build_transition7(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += posi-posj
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    transit = transit + eps

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit



def build_transition8(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for ij in itertools.combinations(range(len(uniques)), 2):
        for col in range(ranks.shape[0]):
            rank = ranks[col,:]
            if uniques[ij[0]] in rank and uniques[ij[1]] in rank:
                posi = np.argwhere(rank==uniques[ij[0]])[0][0]
                posj = np.argwhere(rank==uniques[ij[1]])[0][0]
                transit[ij[1],ij[0]] += posi-posj
    
    
    transit = transit - transit.T
                        
    # no negative proba
    transit[transit<0.] = 0.
    
    transit[transit>0.] = 1.

    # no null columns
    all_zero_columns_index = np.where(~transit.any(axis=0))[0]
    transit[all_zero_columns_index, all_zero_columns_index] = 1.

    # normalizing the columns to one
    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)
    
    # removing dead ends by adding a small probability to jump from any state to any other
    transit = transit + eps

    sums = np.sum(transit, axis=0)
    transit = np.divide(transit, sums)

    # just in case
    transit[transit == np.nan] = 0.
    
    return transit





