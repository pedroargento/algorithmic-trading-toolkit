
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.base import TransformerMixin
import pandas as pd

class AbstractStockSelection(TransformerMixin):
    def stocks(self):
        raise NotImplementedError()
    def weights(self):
        raise NotImplementedError()

class ClusteringSelection(AbstractStockSelection):
    def __init__(self, min_number_of_stocks=20, max_number_of_stocks= 50):
        self.min_number_of_stocks = min_number_of_stocks
        self.max_number_of_stocks = max_number_of_stocks

    def _stocks(self, X):
        max_number_of_stocks = min(self.max_number_of_stocks, X.shape[0]-1)

        best_cluster = KMeans(n_clusters = self.min_number_of_stocks)
        best_cluster.fit(X)
        best_silhouette = metrics.silhouette_score(X, best_cluster.labels_)
    
        for s in range(self.min_number_of_stocks+1, max_number_of_stocks+1):
            km = KMeans(n_clusters = s)
            km.fit(X)
            silhouette = metrics.silhouette_score(X, km.labels_)
            if  silhouette > best_silhouette:
                best_silhouette = silhouette
                best_cluster = km
        
        dist = best_cluster.transform(X)
        df = pd.DataFrame(dist.sum(axis=1), columns=['sqdist'], index = X.index)
        df['cluster'] = best_cluster.labels_
        self.selected_stocks = df.groupby('cluster').sqdist.idxmin() 
        return self.selected_stocks 

    def _weights(self):
        number_of_stocks = len(self.selected_stocks)
        weights = pd.Series(1/number_of_stocks, index = self.selected_stocks ) 
        return weights

    def fit(self, X):
        return self

    def transform(self, X):
        stocks = self._stocks(X)
        weights = self._weights()

        stocks = pd.DataFrame(weights, index=stocks, columns=['weight'])
        stocks.index.rename('stock', inplace=True)
        return stocks


