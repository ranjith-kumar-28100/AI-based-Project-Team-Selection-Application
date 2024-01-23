import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel as sk
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class CollaborativeFilteringRecommender:
    def __init__(self, data):
        self.data = data

    def create_pivot(self, index='Eid', columns=None, values=None):
        return self.data.pivot(index=index, columns=columns, values=values).fillna(0)

    def create_sparse(self, piv):
        self.sparse_data = csr_matrix(piv.values)

    def do_recommendation(self, query, n_neighbors=100):
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(self.sparse_data)
        query = query.reshape(1, -1)
        distances, indices = model.kneighbors(query, n_neighbors=n_neighbors)
        return distances, indices


class ContentBasedRecommender:
    def __init__(self, data):
        self.data = data

    def create_matrix(self, columns):
        df = self.data[columns]
        self.df_matrix = csr_matrix(df.values)

    def create_sigmoid(self):
        self.sig = sk(self.df_matrix, self.df_matrix)

    def do_recommendation(self, eid, top_n=20):
        indices = pd.Series(self.data.index, index=self.data.Eid)
        idx = indices[eid]

        sig_scores = list(enumerate(self.sig[idx]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:(top_n + 1)]  # Get top_n recommendations

        eid_indices = [i[0] for i in sig_scores]
        recommendations = self.data.loc[eid_indices, 'Eid']

        return recommendations
