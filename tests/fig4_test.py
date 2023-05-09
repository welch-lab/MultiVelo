import pytest

import numpy as np
import pandas as pd
import sys
import multivelo as mv
import scanpy as sc
import scvelo as scv
sys.path.append("/../Examples")

# ****** FIG 4 TESTS ******


@pytest.fixture(scope="session")
def result_data_4():

    scv.settings.verbosity = 3
    scv.settings.presenter_view = True
    scv.set_figure_params('scvelo')
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 200)
    np.set_printoptions(suppress=True)

    print("Running fixture!")

    # read in the original AnnData objects
    rna_url = "https://figshare.com/ndownloader/files/40064275"
    atac_url = "https://figshare.com/ndownloader/files/40064278"

    rna_path = "Examples/adata_postpro.h5ad"
    atac_path = "Examples/adata_atac_postpro.h5ad"

    adata_rna = sc.read(rna_path, backup_url=rna_url)
    adata_atac = sc.read(atac_path, backup_url=atac_url)

    # subset genes to run faster
    gene_list = ["Shh", "Heg1", "Cux1", "Lef1"]

    print("Running multivelo")

    # run multivelo

    adata_result = mv.recover_dynamics_chrom(adata_rna,
                                             adata_atac,
                                             gene_list=gene_list,
                                             max_iter=5,
                                             init_mode="invert",
                                             verbose=False,
                                             parallel=True,
                                             n_jobs=15,
                                             save_plot=False,
                                             rna_only=False,
                                             fit=True,
                                             n_anchors=500,
                                             extra_color_key='celltype')

    return adata_result


# the next three tests check to see if recover_dynamics_chrom calculated
# the correct parameters for each of our four genes
def test_alpha(result_data_4):
    alpha = result_data_4.var["fit_alpha"]

    assert alpha[0] == 0.45878197934025416
    assert alpha[1] == 0.08032904996744818
    assert alpha[2] == 1.5346878202804608
    assert alpha[3] == 0.9652887906148591


def test_beta(result_data_4):
    beta = result_data_4.var["fit_beta"]

    assert beta[0] == 0.28770367567423
    assert beta[1] == 0.14497469719573167
    assert beta[2] == 0.564865749852349
    assert beta[3] == 0.2522643118709811


def test_gamma(result_data_4):
    gamma = result_data_4.var["fit_gamma"]

    assert gamma[0] == 0.19648836445315102
    assert gamma[1] == 0.07703610603664116
    assert gamma[2] == 1.0079569101225154
    assert gamma[3] == 0.7485734061079243


# tests the latent_time function
def test_latent_time(result_data_4):

    print("Running latent time")

    mv.velocity_graph(result_data_4)
    mv.latent_time(result_data_4)

    latent_time = result_data_4.obs["latent_time"]

    assert latent_time[0] == 0.248071171005419
    assert latent_time[2000] == 0.21541765304361474
    assert latent_time[4000] == 0.2922034744178431
    assert latent_time[5999] == 0.3094818569923423


# test the velocity_graph function
def test_velo_graph(result_data_4):

    print("Running velocity graph")

    mv.velocity_graph(result_data_4)

    digits = 8

    v_graph_mat = result_data_4.uns["velo_s_norm_graph"].tocoo()

    v_graph = v_graph_mat.data
    v_graph = v_graph.astype(float)
    v_graph = v_graph.round(decimals=digits)

    v_graph_rows = v_graph_mat.row
    v_graph_cols = v_graph_mat.col

    assert len(v_graph) == 1883599
    assert v_graph[0] == 1.0
    assert v_graph[500000] == 1.0
    assert v_graph[1005000] == 0.99999994
    assert v_graph[1500000] == 1.0

    assert v_graph_rows[0] == 0
    assert v_graph_rows[500000] == 1411
    assert v_graph_rows[1005000] == 2834
    assert v_graph_rows[1500000] == 4985

    assert v_graph_cols[0] == 7
    assert v_graph_cols[500000] == 2406
    assert v_graph_cols[1005000] == 2892
    assert v_graph_cols[1500000] == 2480
