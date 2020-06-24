import numpy as np
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve

import pandas as pd
from sklearn.base import TransformerMixin

class AbstractFactorModel(TransformerMixin):
    def transform(self):
        raise NotImplementedError


class PCAFactors(AbstractFactorModel):
    def __init__(self, number_of_factors, **kwargs):
        self.number_of_factors = number_of_factors

    def fit(self, X):
        return self

    def transform (self, X):
        X = X.dropna()
        E, V = eigh(X)
        key = argsort(E)[::-1][:self.number_of_factors]
        E, V = E[key], V[:, key]
        U = dot(X, V)  # used to be dot(V.T, data.T).T
        return pd.DataFrame(U, index = X.index)
