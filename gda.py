import numpy as np

class GDABinaryClassifier:
    
    def fit(self, X, y):
        self.fi = y.mean()
        self.u = np.array([ X[y==k].mean(axis=0) for k in [0,1]])
        X_u = X.copy()
        for k in [0,1]: X_u[y==k] -= self.u[k]
        self.E = X_u.T.dot(X_u) / len(y)
        self.invE = np.linalg.pinv(self.E)
        return self
    
    def predict(self, X):
        return np.argmax([self.compute_prob(X, i) for i in range(len(self.u))], axis=0)
    
    def compute_prob(self, X, i):
        u, phi = self.u[i], ((self.fi)**i * (1 - self.fi)**(1 - i))
        return np.exp(-1.0 * np.sum((X-u).dot(self.invE)*(X-u), axis=1)) * phi
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()