# pyWNN is a package developed by Dylan Kotliar (GitHub username: dylkot), published under the MIT license.

# The original release, including tutorials, can be found here: https://github.com/dylkot/pyWNN

import scanpy as sc
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix, lil_matrix, diags
import time



def get_nearestneighbor(knn, neighbor=1):
    '''For each row of knn, returns the column with the lowest value
    I.e. the nearest neighbor'''
    indices = knn.indices
    indptr = knn.indptr
    data = knn.data
    nn_idx = []
    for i in range(knn.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        nn_idx.append(cols[idx[neighbor-1]])
    return(np.array(nn_idx))


def compute_bw(knn_adj, embedding, n_neighbors=20):
    intersect = knn_adj.dot(knn_adj.T)
    indices = intersect.indices
    indptr = intersect.indptr
    data = intersect.data
    data = data / ((n_neighbors*2) - data)
    bandwidth = []
    for i in range(intersect.shape[0]):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        valssort = rowvals[idx]
        numinset = len(cols)
        if numinset<n_neighbors:
            sys.exit('Fewer than 20 cells with Jacard sim > 0')
        else:
            curval = valssort[n_neighbors]
            for num in range(n_neighbors, numinset):
                if valssort[num]!=curval:
                    break
                else:
                    num+=1
            minjacinset = cols[idx][:num]
            if num <n_neighbors:
                print('shouldnt end up here')
                sys.exit(-1)
            else:
                euc_dist = ((embedding[minjacinset,:]-embedding[i,:])**2).sum(axis=1)**.5
                euc_dist_sorted = np.sort(euc_dist)[::-1]
                bandwidth.append( np.mean(euc_dist_sorted[:n_neighbors]) )
    return(np.array(bandwidth))


def compute_affinity(dist_to_predict, dist_to_nn, bw):
    affinity = dist_to_predict-dist_to_nn
    affinity[affinity<0]=0
    affinity = affinity * -1
    affinity = np.exp(affinity / (bw-dist_to_nn))
    return(affinity)


def dist_from_adj(adjacency, embed1, embed2, nndist1, nndist2):
    dist1 = lil_matrix(adjacency.shape)
    dist2 = lil_matrix(adjacency.shape)

    count = 0
    indices = adjacency.indices
    indptr = adjacency.indptr
    ncells = adjacency.shape[0]

    tic = time.perf_counter()
    for i in range(ncells):
        for j in range(indptr[i], indptr[i+1]):
            col = indices[j]
            a = (((embed1[i,:] - embed1[col,:])**2).sum()**.5) - nndist1[i]
            if a == 0: dist1[i,col] = np.nan
            else: dist1[i,col] = a
            b = (((embed2[i,:] - embed2[col,:])**2).sum()**.5) - nndist2[i]
            if b == 0: dist2[i,col] = np.nan
            else: dist2[i,col] = b

        if (i % 2000) == 0:
            toc = time.perf_counter()
            print('%d out of %d %.2f seconds elapsed' % (i, ncells, toc-tic))

    return(csr_matrix(dist1), csr_matrix(dist2))


def select_topK(dist,  n_neighbors=20):
    indices = dist.indices
    indptr = dist.indptr
    data = dist.data
    nrows = dist.shape[0]

    final_data = []
    final_col_ind = []

    tic = time.perf_counter()
    for i in range(nrows):
        cols = indices[indptr[i]:indptr[i+1]]
        rowvals = data[indptr[i]:indptr[i+1]]
        idx = np.argsort(rowvals)
        final_data.append(rowvals[idx[(-1*n_neighbors):]])
        final_col_ind.append(cols[idx[(-1*n_neighbors):]])

    final_data = np.concatenate(final_data)
    final_col_ind = np.concatenate(final_col_ind)
    final_row_ind = np.tile(np.arange(nrows), (n_neighbors, 1)).reshape(-1, order='F')

    result = csr_matrix((final_data, (final_row_ind, final_col_ind)), shape=(nrows, dist.shape[1]))

    return(result)


class pyWNN():

    def __init__(self, adata, reps=['X_pca', 'X_apca'], n_neighbors=20, npcs=[20, 20], seed=14, distances=None):
        """\
        Class for running weighted nearest neighbors analysis as described in Hao
        et al 2021.
        """

        self.seed = seed
        np.random.seed(seed)

        if len(reps)>2:
            sys.exit('WNN currently only implemented for 2 modalities')

        self.adata = adata.copy()
        self.reps = [r+'_norm' for r in reps]
        self.npcs = npcs
        for (i,r) in enumerate(reps):
            self.adata.obsm[self.reps[i]] = preprocessing.normalize(adata.obsm[r][:,0:npcs[i]])

        self.n_neighbors = n_neighbors
        if distances is None:
            print('Computing KNN distance matrices using default Scanpy implementation')
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=npcs[0], use_rep=self.reps[0], metric='euclidean', key_added='1')
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=npcs[1], use_rep=self.reps[1], metric='euclidean', key_added='2')
            sc.pp.neighbors(self.adata, n_neighbors=200, n_pcs=npcs[0], use_rep=self.reps[0], metric='euclidean', key_added='1_200')
            sc.pp.neighbors(self.adata, n_neighbors=200, n_pcs=npcs[1], use_rep=self.reps[1], metric='euclidean', key_added='2_200')
            self.distances = ['1_distances', '2_distances', '1_200_distances', '2_200_distances']
        else:
            self.distances = distances

        for d in self.distances:
            if type(self.adata.obsp[d]) is not csr_matrix:
                self.adata.obsp[d] = csr_matrix(self.adata.obsp[d])

        self.NNdist = []
        self.NNidx = []
        self.NNadjacency = []
        self.BWs = []

        for (i,r) in enumerate(self.reps):
            nn = get_nearestneighbor(self.adata.obsp[self.distances[i]])
            dist_to_nn = ((self.adata.obsm[r]-self.adata.obsm[r][nn, :])**2).sum(axis=1)**.5
            nn_adj = (self.adata.obsp[self.distances[i]]>0).astype(int)
            nn_adj_wdiag = nn_adj.copy()
            nn_adj_wdiag.setdiag(1)
            bw = compute_bw(nn_adj_wdiag, self.adata.obsm[r], n_neighbors=self.n_neighbors)
            self.NNidx.append(nn)
            self.NNdist.append(dist_to_nn)
            self.NNadjacency.append(nn_adj)
            self.BWs.append(bw)

        self.weights = []
        self.WNN = None

    def compute_weights(self):
        cmap = {0:1, 1:0}
        affinity_ratios = []
        self.within = []
        self.cross = []
        for (i,r) in enumerate(self.reps):
            within_predict = self.NNadjacency[i].dot(self.adata.obsm[r]) / (self.n_neighbors-1)
            cross_predict = self.NNadjacency[cmap[i]].dot(self.adata.obsm[r]) / (self.n_neighbors-1)

            within_predict_dist = ((self.adata.obsm[r] - within_predict)**2).sum(axis=1)**.5
            cross_predict_dist = ((self.adata.obsm[r] - cross_predict)**2).sum(axis=1)**.5
            within_affinity = compute_affinity(within_predict_dist, self.NNdist[i], self.BWs[i])
            cross_affinity = compute_affinity(cross_predict_dist, self.NNdist[i], self.BWs[i])
            affinity_ratios.append(within_affinity / (cross_affinity + 0.0001))
            self.within.append(within_predict_dist)
            self.cross.append(cross_predict_dist)

        self.weights.append( 1 / (1+ np.exp(affinity_ratios[1]-affinity_ratios[0])) )
        self.weights.append( 1 - self.weights[0] )


    def compute_wnn(self, adata):
        print('Computing modality weights')
        self.compute_weights()
        union_adj_mat = ((self.adata.obsp[self.distances[2]]+self.adata.obsp[self.distances[3]]) > 0).astype(int)

        print('Computing weighted distances for union of 200 nearest neighbors between modalities')
        full_dists = dist_from_adj(union_adj_mat, self.adata.obsm[self.reps[0]], self.adata.obsm[self.reps[1]],
                                   self.NNdist[0], self.NNdist[1])
        weighted_dist = csr_matrix(union_adj_mat.shape)
        for (i,dist) in enumerate(full_dists):
            dist = diags(-1 / (self.BWs[i] - self.NNdist[i]), format='csr').dot(dist)
            dist.data = np.exp(dist.data)
            ind = np.isnan(dist.data)
            dist.data[ind] = 1
            dist = diags(self.weights[i]).dot(dist)
            weighted_dist += dist

        print('Selecting top K neighbors')
        self.WNN = select_topK(weighted_dist,  n_neighbors=self.n_neighbors)
        WNNdist = self.WNN.copy()
        x = (1-WNNdist.data) / 2
        x[x<0]=0
        x[x>1]=1
        WNNdist.data = np.sqrt(x)
        self.WNNdist = WNNdist


        adata.obsp['WNN'] = self.WNN
        adata.obsp['WNN_distance'] = self.WNNdist
        adata.obsm[self.reps[0]] = self.adata.obsm[self.reps[0]]
        adata.obsm[self.reps[1]] = self.adata.obsm[self.reps[1]]
        adata.uns['WNN'] = {'connectivities_key': 'WNN',
                                     'distances_key': 'WNN_distance',
                                     'params': {'n_neighbors': self.n_neighbors,
                                      'method': 'WNN',
                                      'random_state': self.seed,
                                      'metric': 'euclidean',
                                      'use_rep': self.reps[0],
                                      'n_pcs': self.npcs[0]}}
        return(adata)
