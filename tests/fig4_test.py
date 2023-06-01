import os
import sys

current_path = os.path.dirname(__file__)
src_path = os.path.join(current_path, "../src/multivelo")
sys.path.append(src_path)

import pytest
import numpy as np
import dynamical_chrom_func as dcf
import scanpy as sc
import scvelo as scv
import sys

sys.path.append("/..")

scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
np.set_printoptions(suppress=True)


rna_path = "test_files/adata_postpro.h5ad"
atac_path = "test_files/adata_atac_postpro.h5ad"


@pytest.fixture(scope="session")
def result_data_4():

    # read in the original AnnData objects
    adata_rna = sc.read(rna_path)
    adata_atac = sc.read(atac_path)

    # subset genes to run faster
    gene_list = ["Shh", "Heg1", "Cux1", "Lef1"]

    # run our first function to test (recover_dynamics_chrom)
    adata_result = dcf.recover_dynamics_chrom(adata_rna,
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

    assert alpha[0] == pytest.approx(0.45878197934025416)
    assert alpha[1] == pytest.approx(0.08032904996744818)
    assert alpha[2] == pytest.approx(1.5346878202804608)
    assert alpha[3] == pytest.approx(0.9652887906148591)


def test_beta(result_data_4):
    beta = result_data_4.var["fit_beta"]

    assert beta[0] == pytest.approx(0.28770367567423)
    assert beta[1] == pytest.approx(0.14497469719573167)
    assert beta[2] == pytest.approx(0.564865749852349)
    assert beta[3] == pytest.approx(0.2522643118709811)


def test_gamma(result_data_4):
    gamma = result_data_4.var["fit_gamma"]

    assert gamma[0] == pytest.approx(0.19648836445315102)
    assert gamma[1] == pytest.approx(0.07703610603664116)
    assert gamma[2] == pytest.approx(1.0079569101225154)
    assert gamma[3] == pytest.approx(0.7485734061079243)


def test_embedding_stream(result_data_4):

    dcf.velocity_graph(result_data_4)

    ax = dcf.velocity_embedding_stream(result_data_4, basis='umap',
                                      color='celltype', show=False)

    assert ax is not None

    assert ax.axis()[0] == pytest.approx(-2.0698418340618714)
    assert ax.axis()[1] == pytest.approx(8.961822542538197)
    assert ax.axis()[2] == pytest.approx(-14.418079041548095)
    assert ax.axis()[3] == pytest.approx(-7.789863798927619)

    assert ax.get_xlim()[0] == pytest.approx(-2.0698418340618714)
    assert ax.get_xlim()[1] == pytest.approx(8.961822542538197)

    assert ax.get_ylim()[0] == pytest.approx(-14.418079041548095)
    assert ax.get_ylim()[1] == pytest.approx(-7.789863798927619)


# tests the latent_time function
def test_latent_time(result_data_4):

    dcf.velocity_graph(result_data_4)
    dcf.latent_time(result_data_4)

    latent_time = result_data_4.obs["latent_time"]

    assert latent_time.shape[0] == 6436


# test the velocity_graph function
def test_velo_graph(result_data_4):

    dcf.velocity_graph(result_data_4)

    digits = 8

    v_graph_mat = result_data_4.uns["velo_s_norm_graph"].tocoo()

    v_graph = v_graph_mat.data
    v_graph = v_graph.astype(float)
    v_graph = v_graph.round(decimals=digits)

    v_graph_rows = v_graph_mat.row
    v_graph_cols = v_graph_mat.col

    assert len(v_graph) == 1883599
    assert v_graph[0] == pytest.approx(1.0)
    assert v_graph[500000] == pytest.approx(1.0)
    assert v_graph[1005000] == pytest.approx(0.99999994)
    assert v_graph[1500000] == pytest.approx(1.0)

    assert v_graph_rows[0] == 0
    assert v_graph_rows[500000] == 1411
    assert v_graph_rows[1005000] == 2834
    assert v_graph_rows[1500000] == 4985

    assert v_graph_cols[0] == 7
    assert v_graph_cols[500000] == 2406
    assert v_graph_cols[1005000] == 2892
    assert v_graph_cols[1500000] == 2480


@pytest.fixture(scope="session")
def lrt_compute():

    # read in the original AnnData objects
    adata_rna = sc.read(rna_path)
    adata_atac = sc.read(atac_path)

    # subset genes to run faster
    gene_list = ["Shh", "Heg1", "Cux1", "Lef1"]

    # run our first function to test (LRT_decoupling)
    w_de, wo_de, res = dcf.LRT_decoupling(adata_rna,
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

    # w_de = with decoupling
    # wo_de = without decoupling
    # res = LRT stats
    return (w_de, wo_de, res)


def decouple_test(lrt_compute):

    w_decouple = lrt_compute[0]

    alpha_c = w_decouple.var["fit_alpha_c"]

    assert alpha_c[0] == pytest.approx(0.057961)
    assert alpha_c[1] == pytest.approx(0.039439)
    assert alpha_c[2] == pytest.approx(0.076731)
    assert alpha_c[3] == pytest.approx(0.063575)

    beta = w_decouple.var["fit_beta"]

    assert beta[0] == pytest.approx(0.287704)
    assert beta[1] == pytest.approx(0.144975)
    assert beta[2] == pytest.approx(0.564866)
    assert beta[3] == pytest.approx(0.252264)

    gamma = w_decouple.var["fit_gamma"]

    assert gamma[0] == pytest.approx(0.196488)
    assert gamma[1] == pytest.approx(0.077036)
    assert gamma[2] == pytest.approx(1.007957)
    assert gamma[3] == pytest.approx(0.748573)


def no_decouple_test(lrt_compute):

    print("No decouple test")

    wo_decouple = lrt_compute[1]

    alpha_c = wo_decouple.var["fit_alpha_c"]

    assert alpha_c[0] == pytest.approx(0.093752)
    assert alpha_c[1] == pytest.approx(0.041792)
    assert alpha_c[2] == pytest.approx(0.051228)
    assert alpha_c[3] == pytest.approx(0.050951)

    beta = wo_decouple.var["fit_beta"]

    assert beta[0] == pytest.approx(0.840938)
    assert beta[1] == pytest.approx(0.182773)
    assert beta[2] == pytest.approx(0.326623)
    assert beta[3] == pytest.approx(0.232073)

    gamma = wo_decouple.var["fit_gamma"]

    assert gamma[0] == pytest.approx(0.561730)
    assert gamma[1] == pytest.approx(0.106799)
    assert gamma[2] == pytest.approx(0.783257)
    assert gamma[3] == pytest.approx(0.705256)


def lrt_res_test(lrt_compute):

    res = lrt_compute[2]

    likelihood_c_w_decoupled = res["likelihood_c_w_decoupled"]

    assert likelihood_c_w_decoupled[0] == pytest.approx(0.279303)
    assert likelihood_c_w_decoupled[1] == pytest.approx(0.186213)
    assert likelihood_c_w_decoupled[2] == pytest.approx(0.295591)
    assert likelihood_c_w_decoupled[3] == pytest.approx(0.144158)

    likelihood_c_wo_decoupled = res["likelihood_c_wo_decoupled"]

    assert likelihood_c_wo_decoupled[0] == pytest.approx(0.270491)
    assert likelihood_c_wo_decoupled[1] == pytest.approx(0.180695)
    assert likelihood_c_wo_decoupled[2] == pytest.approx(0.294631)
    assert likelihood_c_wo_decoupled[3] == pytest.approx(0.175622)

    LRT_c = res["LRT_c"]

    assert LRT_c[0] == pytest.approx(412.637730)
    assert LRT_c[1] == pytest.approx(387.177688)
    assert LRT_c[2] == pytest.approx(41.850304)
    assert LRT_c[3] == pytest.approx(-2541.289231)

    pval_c = res["pval_c"]

    assert pval_c[0] == pytest.approx(9.771580e-92)
    assert pval_c[1] == pytest.approx(3.406463e-86)
    assert pval_c[2] == pytest.approx(9.853544e-11)
    assert pval_c[3] == pytest.approx(1.000000e+00)

    likelihood_w_decoupled = res["likelihood_w_decoupled"]

    assert likelihood_w_decoupled[0] == pytest.approx(0.177979)
    assert likelihood_w_decoupled[1] == pytest.approx(0.008453)
    assert likelihood_w_decoupled[2] == pytest.approx(0.140156)
    assert likelihood_w_decoupled[3] == pytest.approx(0.005029)

    likelihood_wo_decoupled = res["likelihood_wo_decoupled"]

    assert likelihood_wo_decoupled[0] == pytest.approx(0.181317)
    assert likelihood_wo_decoupled[1] == pytest.approx(0.009486)
    assert likelihood_wo_decoupled[2] == pytest.approx(0.141367)
    assert likelihood_wo_decoupled[3] == pytest.approx(0.008299)

    LRT = res["LRT"]

    assert LRT[0] == pytest.approx(-239.217562)
    assert LRT[1] == pytest.approx(-1485.199859)
    assert LRT[2] == pytest.approx(-110.788912)
    assert LRT[3] == pytest.approx(-6447.599212)

    pval = res["pval"]

    assert pval[0] == pytest.approx(1.0)
    assert pval[1] == pytest.approx(1.0)
    assert pval[2] == pytest.approx(1.0)
    assert pval[3] == pytest.approx(1.0)

    c_likelihood = res["likelihood_c_w_decoupled"]

    assert c_likelihood[0] == pytest.approx(0.279303)
    assert c_likelihood[1] == pytest.approx(0.186213)
    assert c_likelihood[2] == pytest.approx(0.295591)
    assert c_likelihood[3] == pytest.approx(0.144158)

    def test_qc_metrics():
        adata_rna = sc.read(rna_path)

        dcf.calculate_qc_metrics(adata_rna)

        total_unspliced = adata_rna.obs["total_unspliced"]

        assert total_unspliced.shape == (6436,)
        assert total_unspliced[0] == pytest.approx(91.709404)
        assert total_unspliced[1500] == pytest.approx(115.21283)
        assert total_unspliced[3000] == pytest.approx(61.402004)
        assert total_unspliced[4500] == pytest.approx(84.03409)
        assert total_unspliced[6000] == pytest.approx(61.26761)

        total_spliced = adata_rna.obs["total_spliced"]

        assert total_spliced.shape == (6436,)
        assert total_spliced[0] == pytest.approx(91.514175)
        assert total_spliced[1500] == pytest.approx(66.045616)
        assert total_spliced[3000] == pytest.approx(87.05275)
        assert total_spliced[4500] == pytest.approx(83.82857)
        assert total_spliced[6000] == pytest.approx(62.019516)

        unspliced_ratio = adata_rna.obs["unspliced_ratio"]

        assert unspliced_ratio.shape == (6436,)
        assert unspliced_ratio[0] == pytest.approx(0.5005328)
        assert unspliced_ratio[1500] == pytest.approx(0.6356273)
        assert unspliced_ratio[3000] == pytest.approx(0.4136075)
        assert unspliced_ratio[4500] == pytest.approx(0.50061214)
        assert unspliced_ratio[6000] == pytest.approx(0.4969506)

        cell_cycle_score = adata_rna.obs["cell_cycle_score"]

        assert cell_cycle_score.shape == (6436,)
        assert cell_cycle_score[0] == pytest.approx(-0.24967776384597046)
        assert cell_cycle_score[1500] == pytest.approx(0.5859756395543293)
        assert cell_cycle_score[3000] == pytest.approx(0.06501555292615813)
        assert cell_cycle_score[4500] == pytest.approx(0.1406775909466575)
        assert cell_cycle_score[6000] == pytest.approx(-0.33825528386759895)
