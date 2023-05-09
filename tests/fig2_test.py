import pytest

import numpy as np
import pandas as pd
import sys
import multivelo as mv
import scanpy as sc
import scvelo as scv
import random
import math

sys.path.append("/..")

scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
np.set_printoptions(suppress=True)


@pytest.fixture(scope="session")
def result_data_2():

    # read in the raw data
    adata_atac = sc.read_10x_mtx('Examples/outs/filtered_feature_bc_matrix/',
                                 var_names='gene_symbols', cache=True,
                                 gex_only=False)
    adata_atac = adata_atac[:, adata_atac.var['feature_types'] == "Peaks"]

    # subset the data to run the test faster
    N = adata_atac.shape[1]
    n_sub = math.floor(0.1*N)
    random.seed(42)
    choices = random.choices(range(N), k=n_sub)
    adata_atac = adata_atac[:, choices]

    # aggregate peaks
    adata_atac = mv.aggregate_peaks_10x(adata_atac,
                                        'Examples/outs/peak_annotation.tsv',
                                        'Examples/outs/analysis/feature_linkage/feature_linkage.bedpe',
                                        verbose=False)

    return adata_atac


# test the aggregate_peaks_10x function
def test_agg_peaks(result_data_2):

    # the data indices we want to check
    indices = [0, 500000, 1000000, 1500000, 2000000]

    # the results we should get
    data = [8.0, 2.0, 2.0, 2.0, 4.0]
    rows = [0, 1072, 2149, 3247, 4342]
    cols = [9, 15, 1645, 1156, 1633]

    # convert the atac data into coo form
    atac_coo = result_data_2.X.tocoo()

    # make sure that 
    assert len(atac_coo.data) == 2241632

    # make sure that the data, rows, and columns all match
    for n, i in enumerate(indices):

        assert atac_coo.data[i] == pytest.approx(data[n])
        assert atac_coo.row[i] == rows[n]
        assert atac_coo.col[i] == cols[n]


def test_tfidf(result_data_2):

    tfidf_result = result_data_2.copy()

    # run tfidf
    mv.tfidf_norm(tfidf_result)

    # the data indices we want to check
    indices = [0, 500000, 1000000, 1500000, 2000000]

    # the results we should get
    data = [12.598012, 4.446732, 10.223078, 14.305816, 21.292938]
    rows = np.array([0, 1072, 2149, 3247, 4342])
    cols = np.array([9, 15, 1645, 1156, 1633])

    # convert the atac data into coo form
    atac_coo = tfidf_result.X.tocoo()

    # make sure that the length of the data array is correct
    assert len(atac_coo.data) == 2241632

    # make sure that the data, rows, and columns all match
    for n, i in enumerate(indices):

        assert atac_coo.data[i] == pytest.approx(data[n])
        assert atac_coo.row[i] == rows[n]
        assert atac_coo.col[i] == cols[n]


def test_smooth(result_data_2):

    new_result = result_data_2.copy()

    # load in the smoothing matrices
    nn_idx = np.loadtxt("Examples/seurat_wnn/nn_idx.txt", delimiter=',')
    nn_dist = np.loadtxt("Examples/seurat_wnn/nn_dist.txt", delimiter=',')

    # subset the ATAC data to make sure we can use the matrices
    atac_smooth = new_result[:nn_idx.shape[0], :]

    # run knn_smooth_chrom
    mv.knn_smooth_chrom(atac_smooth, nn_idx, nn_dist)

    # the data indices we want to check
    indices = [0, 500000, 1000000, 1500000, 1552093]

    # the results we should get
    data = [8.0, 2.0, 2.0, 2.0, 2.0]
    rows = [0, 1072, 2149, 3247, 3364]
    cols = [9, 15, 1645, 1156, 2008]

    # convert the atac data into coo form
    atac_coo = atac_smooth.X.tocoo()

    # make sure that the length of the data array is correct
    assert len(atac_coo.data) == 1552094

    # make sure that the data, rows, and columns all match
    for n, i in enumerate(indices):

        assert atac_coo.data[i] == pytest.approx(data[n])
        assert atac_coo.row[i] == rows[n]
        assert atac_coo.col[i] == cols[n]
