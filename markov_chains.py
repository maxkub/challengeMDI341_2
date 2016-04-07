import numpy as np

def find_label(ranks, probe_id, eps=1e-2, tol=1e-3, max_rounds=500):
    """
    Function that return the most probable label based on a Markov chain algorithm
    """
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)
    
    # build transition matrix
    transit = build_transition1(ranks, eps)
    
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
        #diff = np.abs(np.linalg.norm(label_proba2) - np.linalg.norm(label_proba1))
        i += 1
        
    return label_proba2

def new_rankings(ranks, probe_id, eps=1e-1, tol=1e-5, max_rounds=100, seed=42):
    """
    Combining multiple rankings based on markov chains, with different transition matrix
    """
    np.random.seed(seed)
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)
    
    # build transition matrix
    transit1 = build_transition1(ranks, eps=np.random.uniform(0.3,0.8))
    transit2 = build_transition2(ranks, eps=np.random.uniform(0.3,0.8))
    transit3 = build_transition3(ranks, eps=np.random.uniform(0.01,0.5))
    transit4 = build_transition4(ranks, eps=np.random.uniform(0.01,0.5))
    transit5 = build_transition5(ranks, eps=np.random.uniform(0.3,0.8))
    transit6 = build_transition6(ranks, eps=np.random.uniform(0.3,0.8))
    transit7 = build_transition7(ranks, eps=np.random.uniform(0.01,0.5))
    transit8 = build_transition8(ranks, eps=np.random.uniform(0.01,0.5))
    
    transit = np.stack((transit1, transit2, transit3, transit4, transit5, transit6, transit7, transit8))
    
    print 'big matrix:', transit.shape
    
    rankings = np.empty([ranks.shape[1], transit.shape[0]])
    for i in range(transit.shape[0]):
        res = get_equilibrium(transit[i,:,:], uniques, tol, max_rounds)
        ind = np.argsort(-res)
        rankings[:,i] = uniques[ind][:ranks.shape[1]]
    
    return rankings
    

def build_transition1(ranks, eps):
    """
    Fonction to build the transition matrix between all possible states of the Markov chain
    """
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.#*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.#*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.#*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.#*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.#*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.#*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.#*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.#*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.*(posj-posi)
                        
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
    
    ranks = ranks.T
    
    # finding the possible states of the markov chain
    uniques = np.unique(ranks)

    # initializing the transition matrix
    transit = np.zeros((len(uniques), len(uniques)))

    # filling the transition matrix with transition probabilities between all possible states
    for i in range(len(uniques)):
        for j in range(len(uniques)):
            for col in range(8):
                rank = ranks[:,col]
                if i!=j and uniques[i] in rank and uniques[j] in rank:
                    posi = np.argwhere(rank==uniques[i])[0][0]
                    posj = np.argwhere(rank==uniques[j])[0][0]
                    if posi > posj:
                        transit[j,i] += 1.*(posi-posj)
                    elif posi < posj:
                        transit[j,i] -= 1.*(posj-posi)
                        
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




