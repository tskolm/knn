import collections
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import numpy as np

# In[145]:
class KNN_classifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.prev_k = 0
        self.strategy = strategy
        if metric == None:          
            self.metric = 'euclidean'
        else:
            self.metric = metric
        if weights == None:
            self.weights = False
        else:
            self.weights = weights
        self.test_block_size = test_block_size
        self.ans = None
        self.model = None
        
    def fit(self, X, y=None):
        if self.strategy == 'my_own':
            self.model = X
        else:
            self.model = NearestNeighbors(self.k, metric=self.metric, algorithm=self.strategy)
            self.model.fit(X, y)
        self.ans = y
            
    def find_kneighbors(self, X, k=None):   
        if k == None:
            k = self.k
        if self.strategy == 'my_own':
            fitted = distance.cdist(X, self.model, self.metric) 
            nbrs = []
            dist = []
            for i in range(fitted.shape[0]):
                nbrs.append(np.argsort(fitted[i])[:k])
                dist.append(np.sort(fitted[i])[:k])
            if self.weights:
                return (np.array(dist), np.array(nbrs))
            else:
                return np.array(nbrs)
        else:
            if self.weights:
                return self.model.kneighbors(X, return_distance=self.weights)
            else:
                return self.model.kneighbors(X, return_distance=self.weights)[:, :k]
    
    def interval_prediction(self, X, k=None):
        if k == None:
            k = self.k
        eps = 1e-5
        predict = []
        if self.weights:
            dist, nbrs = self.find_kneighbors(X, k)
            for i in range(nbrs.shape[0]):
                classes = dict()
                for j in range(k):
                    weight = 1.0 / (dist[i][j] + eps)
                    if not (self.ans[nbrs[i][j]] in classes.keys()):
                        classes[self.ans[nbrs[i][j]]] = weight
                    else:                            
                        classes[self.ans[nbrs[i][j]]] += weight

                sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse = True)
                predict.append(sorted_classes[0][0])                    
        else:
            nbrs = self.find_kneighbors(X, k)
            for i in range(nbrs.shape[0]):   
                predict.append(collections.Counter(self.ans[nbrs[i]]).most_common()[0][0])  
        return predict
    
    def predict(self, X, k=None): 
        if k == None:
            k = self.k
        if self.test_block_size == None:
            self.test_block_size = X.shape[0]
        predict = np.array([])
        for i in range(int(X.shape[0]/self.test_block_size)):  
            predict = np.append(predict, self.interval_prediction(X[i*self.test_block_size:(i+1)*self.test_block_size], k))
        size = int(X.shape[0]/self.test_block_size)*self.test_block_size
        if size != X.shape[0]:
            predict = np.append(predict, self.interval_prediction(X[size:X.shape[0]], k)) 
        return np.array(predict.astype(int))
