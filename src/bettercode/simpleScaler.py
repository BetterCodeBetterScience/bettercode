# custom scaler class for testing example

class SimpleScaler:
    def __init__(self):
        self.transformed_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X):
        self.transformed_ = (X - self.mean_) / self.std_
        return self.transformed_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

