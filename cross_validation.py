import numpy as np
import knn

def kfold(n, n_folds):
    rand = np.random.permutation(n)
    perm = []
    size = int(n/n_folds)
    for i in range(n_folds-1):
        perm.append(rand[i*size:(i+1)*size])
    perm.append(rand[(n_folds-1)*size: n])
    ret = []
    for i in range(n_folds):
        cur_array = np.array([])
        for j in range(n_folds):
            if i != j:
                cur_array = np.append(cur_array, perm[j])
        cur_array = cur_array.astype(int)
        tup = (cur_array, perm[i])
        ret.append(tup)
    return ret


def knn_cross_val_score(X, y, k_list, score, cv, weights, metric, strategy='brute', test_block_size=None):
    if cv == None:
        cv = kfold(X.shape[0], 3)
    if k_list == None:
        k_list = np.array([1])
    total_accuracy = dict()
    max_k = k_list[-1]   
    classifier = knn.KNN_classifier(max_k, strategy, metric, weights, test_block_size)  
    for k in k_list:    
        if not (k in total_accuracy):
            total_accuracy[k] = []
        for train, test in cv:   
            classifier.fit(X[train], y[train]) 
            accuracy = accuracy_score(classifier.predict(X[test], k), y[test])
            total_accuracy[k].append(accuracy)
    return total_accuracy

def accuracy_score(a, b):
    return float(np.sum(a == b) / np.size(b))