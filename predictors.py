import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm.notebook import tqdm

def rank(array,num):
        """Find rank of number against an array"""
        # much more efficient way
        return np.sum(array>=num)+1

def p_values(post_conform_scores, valid_conform_scores):
    '''Calculates the smooth p-value of a non-conformity score against an array
        of nonconformity scores'''
    # preload to store values
    table = np.ones(post_conform_scores.shape)
    for i, row in enumerate(post_conform_scores):
        for label, conform_score in enumerate(row):
            # get rank
            smooth = np.random.uniform()*np.sum(valid_conform_scores==conform_score)
            table[i,label] = rank(valid_conform_scores, conform_score) + smooth # no need for +1 since count starts at 1
    # divide table by size of validation set + 1 (to include the postulated sample)
    table = table/(len(valid_conform_scores)+1)
    return table

### Nearest neighbour conformal predictors

def Conform_diff_same(distances, label, y_labels):
    """
    Applies the conformal measure diffdist/samedist on distances
    Input:  distances - is an nx1 array containing distances from training set
    Output: diffdist/samedist
            where diffdist is the distance to the nearest sample of a different class
            and samedist is the distance to the nearest sample of the same class
    """
    tempsame = distances[y_labels == label]   # create array containing all distances against samples with same class
    tempdiff = distances[y_labels != label]   # create array containing all distances against samples with diff class
    samedist = np.min(tempsame)   # grab min dist from training samples with same label
    diffdist = np.min(tempdiff)   # grab min dist from training samples with diff label
    if samedist == 0 and diffdist == 0:
        return 0
    elif samedist == 0 and diffdist !=0:
        return np.inf
    else:
        return samedist/diffdist

def NNpvalues_full(X_train, y_train, X_test, y_test):
    '''Transductive conformal predictor using 1-nearest neighbour'''
    labels = np.unique(y_train) # get labels of training data
    rows = len(y_train) # number of training samples
    train_conf_scores = np.zeros(rows) # preload array
    p_values = np.zeros((X_test.shape[0], len(labels))) # preload matrix of p-values
    indices = np.arange(rows) # to easily change diagonal of distances
    train_distances = np.array(pairwise_distances(X_train, n_jobs=-1)) # get training distances
    train_distances[indices,indices] = np.inf # make the diagonal infinity because they will all be 0 and we don't want to consider them
    test_distances = np.array(pairwise_distances(X_test, X_train, n_jobs=-1)) # get distances against training samples for each test sample
    # get conformity scores of training samples
    # don't need to cycle through each label for training
    for i, dist in tqdm(enumerate(train_distances)):
        train_conf_scores[i] = Conform_diff_same(dist, y_train[i], y_train) # get conformity score for sample given it's correct label
    # get conformity scores of test samples
    for i, dist in tqdm(enumerate(test_distances)):
        # cycle through each postulated label
        for j, label in enumerate(labels):
            conf_score = Conform_diff_same(dist, j, y_train) # calculate conformity score
            smooth = np.random.uniform()*np.sum(train_conf_scores==conf_score)
            p_values[i,j] = rank(train_conf_scores, conf_score)+smooth # calculate the rank here
    p_values = p_values/(len(y_train)+1) # need to divide by (# of training sample +1) to update to p-values
    return p_values

def NNpvalues_online(X_data, y_data, labels):
    '''On-line conformal predictor using 1-nearest neighbour'''
    rows = len(y_data) # get total number of samples
    train_conf_scores = np.zeros(rows) # preload array
    p_values = np.zeros((X_data.shape[0], len(labels))) # preload matrix
    print('Step 1: Getting distances')
    all_distances = np.array(pairwise_distances(X_data, n_jobs=-1)) # get all distances now so we don't have to repeat
    all_distances[np.arange(rows),np.arange(rows)] = np.inf # change diagonals to infinity since they will all be 0
    print('Completed')
    seen_labels = [] # as we see each label we will append to this since we can't calculate non-conformity scores if the label is unseen
    print('Step 2: Getting non-conformity scores')
    for k in tqdm(range(1,rows)): # start at 1 since we want at least one training sample to begin with
        train_rows = k # number of training samples
        if y_data[k] not in seen_labels:
            seen_labels.append(y_data[k]) # add new label to our list of labels
        # create new variables to make the code easier to read
        train_distances = all_distances[:k,:k] # our training samples include all distances up to kth row and column
        train_labels = y_data[:k] # our training labels
        train_conf_scores = np.zeros(train_rows) # preload array to save non-conformity scores
        test_distances = all_distances[:k+1,k] # our test distances have one extra row
        for j, dist in enumerate(train_distances): # cycle through each training sample
            if y_data[j] in seen_labels and len(seen_labels)>1: # to make sure we have at least one label the same and one different
                train_conf_scores[j] = Conform_diff_same(dist, train_labels[j], train_labels) # get non-conformity score for correct label
            #if j==train_distances.shape[0]-1:
                #print(train_conf_scores)
        for label in seen_labels: # only need to cycle through seen labels
            if len(seen_labels)>1: # need at least one different label
                conf_score = Conform_diff_same(test_distances, label, y_data[:k+1]) # get non-conformity score
                smooth = np.random.uniform()*np.sum(train_conf_scores==conf_score)
                p_values[k,label] = (rank(train_conf_scores, conf_score)+smooth)/(len(train_labels)+1) # calculate p-values here since number of training samples is continuously changing
    print('Completed')
    return p_values

def NNpvalues_induct(X_train, y_train, X_valid, y_valid, X_test, y_test):
    '''Inductive conformal predictor using 1-nearest neighbour'''
    labels = np.unique(y_train) # get labels of training data
    rows = len(y_train) # number of training samples
    valid_conf_scores = np.zeros(rows) # preload array
    p_values = np.zeros((X_test.shape[0], len(labels))) # preload matrix of p-values
    indices = np.arange(rows) # to easily change diagonal of distances
    valid_distances = np.array(pairwise_distances(X_valid, X_train, n_jobs=-1)) # get training distances
    test_distances = np.array(pairwise_distances(X_test, X_train, n_jobs=-1)) # get distances against training samples for each test sample
    # get conformity scores of validation samples
    # don't need to cycle through each label for training
    for i, dist in tqdm(enumerate(valid_distances)):
        valid_conf_scores[i] = Conform_diff_same(dist, y_valid[i], y_train) # get conformity score for sample given it's correct label
    # get conformity scores of test samples
    for i, dist in tqdm(enumerate(test_distances)):
        # cycle through each postulated label
        for j, label in enumerate(labels):
            conf_score = Conform_diff_same(dist, j, y_train) # calculate conformity score
            smooth = np.random.uniform()*np.sum(valid_conf_scores==conf_score)
            p_values[i,j] = rank(valid_conf_scores, conf_score)+smooth # calculate the rank here

    p_values = p_values/(len(y_valid)+1) # need to divide by (# of training sample +1) to update to p-values
    return p_values

### SKLearn inductive conformal predictors

def inductive_conform_predictor(X_train, y_train, X_valid, y_valid, X_test, y_test, model):
    '''Inductive CP for SKLearn models'''
    model.fit(X_train, y_train)
    valid_conform_scores = np.array(model.predict_proba(X_valid))
    valid_conform_scores = -1 * np.array([valid_conform_scores[i,label] for i, label in enumerate(y_valid)])
    probs = model.predict_proba(X_test)
    post_conform_scores = -1 * probs
    p_vals = p_values(post_conform_scores,valid_conform_scores)
    return p_vals

### Evaluation of CPs

def conf_cred(p_values, y_test):
    '''
    Outputs nx3 table
    1st column contains true lable
    2nd column contains the point prediction
    3rd column contains the confidence of the prediction
    4th column contains the credibility of the prediction
    '''
    # sort the p_values with 2 largest at the end (efficient sort)
    idx = np.argpartition(p_values,-2, axis=1)
    # preload to store
    table = np.zeros((p_values.shape[0],3))
    for i, row in enumerate(p_values):
        point_pred = idx[i,-1] # get point prediction
        cred = p_values[i,idx[i,-1]] # get credibility value for prediction
        conf = 1-p_values[i,idx[i,-2]] # get confidence value for prediction
        table[i] = [point_pred, conf, cred]
    return np.column_stack((y_test,table))

### Anomaly detection

def conform_predictor_model(X_valid, y_valid, X_test, model):
    '''Calculates the p-values of test samples against a validation set on a pre trained model'''
    valid_conform_scores = np.array(model.predict_proba(X_valid))
    valid_conform_scores = -1 * np.array([valid_conform_scores[i,label] for i, label in enumerate(y_valid)])
    post_conform_scores = -1 * model.predict_proba(X_test)
    p_vals = p_values(post_conform_scores,valid_conform_scores)
    return p_vals
