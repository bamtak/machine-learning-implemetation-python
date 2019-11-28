import random
import math

def Kfold(X, folds=5, shuffle=False):
    X = list(range(X)) if isinstance(X, int) else list(range(len(X)))
    if shuffle: random.shuffle(X)
    i = 0
    fold_size = int(math.ceil(1.0 * len(X)/folds))
    while i < folds:
        yield X[0:fold_size*i] + X[fold_size*(i + 1):], X[fold_size*i:fold_size*(i + 1)]
        i += 1
        
def train_test_split(X, y, test_size=0.2, shuffle=True):
    idx = range(len(y))
    if shuffle: random.shuffle(idx)
    test_idx = int(test_size * len(y))
    return X[idx[test_idx:]], X[idx[:test_idx]], y[idx[test_idx:]], y[idx[:test_idx]]

def accuracy_score(pre, y):
    return 1 - sum(1.0 * (pre - y)**2)/len(y)