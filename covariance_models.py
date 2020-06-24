import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

class AbstractCovarianceModel(TransformerMixin):
    def compute_covariance(self):
        raise NotImplementedError

class ConstantCovarianceModel(AbstractCovarianceModel):
    def fit(self, X):
        return self

    def transform(self, returns):
        sample_correlation_matrix = returns.corr()
        n = sample_correlation_matrix.shape[0]
        mean_correlation = (sample_correlation_matrix.values.sum()-n)/(n*(n-1))
        constant_correlation_matrix = np.full_like(sample_correlation_matrix, mean_correlation)
        np.fill_diagonal(constant_correlation_matrix, 1.)
        standard_deviation_vector = returns.std()
        covariance_matrix = constant_correlation_matrix * np.outer(standard_deviation_vector, standard_deviation_vector)
        return pd.DataFrame(covariance_matrix, index=returns.columns, columns=returns.columns)


class SampleCovariance(AbstractCovarianceModel):
    def transform(self, returns, **kwargs):
        return returns.cov()

    def fit(self, X):
        return self


class ShrinkageCovariance(AbstractCovarianceModel):
    def __init__(self, model, delta=0.5, **kwargs):
        self.delta = delta
        self.model = model(**kwargs)

    def transform (self, returns):
        model_covariance = self.model.transform(returns)
        sample_covariance = returns.cov()
        return self.delta*model_covariance + (1-self.delta)*sample_covariance 
    
    def fit(self, X):
        return self


