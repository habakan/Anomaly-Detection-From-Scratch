import numpy as np
from scipy import stats

class Mahalanobis:

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.cov(X.T)
        self.threshold = stats.chi2.interval(0.99, 1)[1]

    def predict(self, X):
        anom_score = []
        for x in X:
            score =  np.dot(np.dot((x-self.mu).T, np.linalg.inv(self.sigma)), x-self.mu)
            anom_score.append(score)
        return [1 if score > self.threshold else 0 for score in anom_score]
        
