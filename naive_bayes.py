import numpy as np

class NaiveBayesBinaryClassifier:
    
    def fit(self, X, y):
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.phi_y = 1.0 * y_counts/y_counts.sum()
        self.phi_x = [1.0 * X[y==k].sum(axis=0)/np.sum(y==k) for k in self.y_classes]
        return self
    
    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]
    
    def compute_prob(self, x, y):
        res = 1
        for j in range(len(x)):
            Pxy = self.phi_x[y][j] # p(xj=1|y)
            res *= (Pxy**x[j])*((1-Pxy)**(1-x[j])) # p(xj=0|y)
        return res * self.phi_y[y]
    
    def evaluate(self, X, y):
        return (self.predict(X) == y).mean()
    
class NaiveBayes:
    
    def fit(self, X, y):
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.x_classes = np.array([np.unique(x) for x in X.T])
        self.phi_y = 1.0 * y_counts/y_counts.sum()
        self.phi_x = [[[(1.0 * (X[:,j][y==k] == i)).sum()/np.sum(y==k) 
                        for i in self.x_classes[j]] 
                       for j in range(len(self.x_classes))] 
                      for k in self.y_classes]
        return self
    
    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]
    
    def compute_prob(self, x, y):
        Pxy = 1
        for j in range(len(x)):
            idx_j = list(self.x_classes[j]).index(x[j])
            Pxy *= self.phi_x[y][j][idx_j] # p(xj|y)
        return Pxy * self.phi_y[y]
    
    def evaluate(self, X, y):
        return (self.predict(X) == y).mean()


if __name__ == '__main__':
    print('Hello')
    
