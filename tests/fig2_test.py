import os
import sys

current_path = os.path.dirname(__file__)
src_path = os.path.join(current_path, "../src/multivelo")
sys.path.append(src_path)

import pytest
import numpy as np
import sys
import auxiliary as a
import scanpy as sc
import scvelo as scv

scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
np.set_printoptions(suppress=True)


@pytest.fixture(scope="session")
def result_data_2():

    # read in the data
    adata_atac = sc.read("test_files/fig2_for_test.h5ad")

    # aggregate peaks
    adata_atac = a.aggregate_peaks_10x(adata_atac,
                                       'test_files/peak_annotation.tsv',
                                       'test_files/feature_linkage.bedpe')

    return adata_atac


# test the aggregate_peaks_10x function
def test_agg_peaks(result_data_2):

    # the data indices we want to check
    indices = [0, 100000, 200000, 300000, 400000]

    # the results we should get
    data = [8.0, 4.0, 2.0, 2.0, 2.0]
    rows = [0, 1157, 2333, 3531, 4724]
    cols = [9, 276, 291, 78, 201]

    # convert the atac data into coo form
    atac_coo = result_data_2.X.tocoo()

    # check that there are the expected number of datapoints
    assert len(atac_coo.data) == 412887

    # make sure that the data, rows, and columns all match
    for n, i in enumerate(indices):

        assert atac_coo.data[i] == pytest.approx(data[n])
        assert atac_coo.row[i] == rows[n]
        assert atac_coo.col[i] == cols[n]


def test_tfidf(result_data_2):

    tfidf_result = result_data_2.copy()

    # run tfidf
    a.tfidf_norm(tfidf_result)

    # the data indices we want to check
    indices = [0, 100000, 200000, 300000, 400000]

    # the results we should get
    data = [66.66449, 29.424345, 85.36392, 42.239613, 26.855387]
    rows = np.array([0, 1157, 2333, 3531, 4724])
    cols = np.array([9, 276, 291, 78, 201])

    # convert the atac data into coo form
    atac_coo = tfidf_result.X.tocoo()

    # make sure that the length of the data array is correct
    assert len(atac_coo.data) == 412887

    # make sure that the data, rows, and columns all match
    for n, i in enumerate(indices):

        assert atac_coo.data[i] == pytest.approx(data[n])
        assert atac_coo.row[i] == rows[n]
        assert atac_coo.col[i] == cols[n]


def test_smooth(result_data_2):

    new_result = result_data_2.copy()

    # load in the smoothing matrices
    nn_idx = np.loadtxt("test_files/nn_idx.txt", delimiter=',')
    nn_dist = np.loadtxt("test_files/nn_dist.txt", delimiter=',')

    # subset the ATAC data to make sure we can use the matrices
    atac_smooth = new_result[:nn_idx.shape[0], :]

    # run knn_smooth_chrom
    a.knn_smooth_chrom(atac_smooth, nn_idx, nn_dist)

    # the data indices we want to check
    indices = [0, 70000, 140000, 210000, 280000]

    # the results we should get
    data = [8.0, 4.0, 2.0, 2.0, 2.0]
    rows = [0, 809, 1615, 2453, 3295]
    cols = [9, 327, 56, 25, 137]

    # convert the atac data into coo form
    atac_coo = atac_smooth.X.tocoo()

    # make sure that the length of the data array is correct
    assert len(atac_coo.data) == 285810

    # make sure that the data, rows, and columns all match
    for n, i in enumerate(indices):

        assert atac_coo.data[i] == pytest.approx(data[n])
        assert atac_coo.row[i] == rows[n]
        assert atac_coo.col[i] == cols[n]
