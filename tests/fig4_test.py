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
import auxiliary as a
import math

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
    assert latent_time[0] == pytest.approx(0.13637730355232341)
    assert latent_time[2000] == pytest.approx(0.23736570450245578)
    assert latent_time[4000] == pytest.approx(0.27272782940968826)
    assert latent_time[5999] == pytest.approx(0.5737945319304785)


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


def test_decouple(lrt_compute):

    w_decouple = lrt_compute[0]

    alpha_c = w_decouple.var["fit_alpha_c"]

    assert alpha_c[0] == pytest.approx(0.05796053555915494)
    assert alpha_c[1] == pytest.approx(0.03943890374543767)
    assert alpha_c[2] == pytest.approx(0.0767312432562168)
    assert alpha_c[3] == pytest.approx(0.06357529403223793)

    beta = w_decouple.var["fit_beta"]

    assert beta[0] == pytest.approx(0.28770367567423)
    assert beta[1] == pytest.approx(0.14497469719573167)
    assert beta[2] == pytest.approx(0.564865749852349)
    assert beta[3] == pytest.approx(0.2522643118709811)

    gamma = w_decouple.var["fit_gamma"]

    assert gamma[0] == pytest.approx(0.19648836445315102)
    assert gamma[1] == pytest.approx(0.07703610603664116)
    assert gamma[2] == pytest.approx(1.0079569101225154)
    assert gamma[3] == pytest.approx(0.7485734061079252)


def test_no_decouple(lrt_compute):

    print("No decouple test")

    wo_decouple = lrt_compute[1]

    alpha_c = wo_decouple.var["fit_alpha_c"]

    assert alpha_c[0] == pytest.approx(0.09375182014995963)
    assert alpha_c[1] == pytest.approx(0.04179216577260276)
    assert alpha_c[2] == pytest.approx(0.051227904137383304)
    assert alpha_c[3] == pytest.approx(0.05095112785526705)

    beta = wo_decouple.var["fit_beta"]

    assert beta[0] == pytest.approx(0.8409379018031923)
    assert beta[1] == pytest.approx(0.18277265733758338)
    assert beta[2] == pytest.approx(0.3266225295086788)
    assert beta[3] == pytest.approx(0.23207288362156378)

    gamma = wo_decouple.var["fit_gamma"]

    assert gamma[0] == pytest.approx(0.5617299429106162)
    assert gamma[1] == pytest.approx(0.10679919937554674)
    assert gamma[2] == pytest.approx(0.7832567821244719)
    assert gamma[3] == pytest.approx(0.7052561318884707)


def test_lrt_res(lrt_compute):

    res = lrt_compute[2]

    likelihood_c_w_decoupled = res["likelihood_c_w_decoupled"]

    assert likelihood_c_w_decoupled[0] == pytest.approx(0.27930299376829615)
    assert likelihood_c_w_decoupled[1] == pytest.approx(0.1862125750499589)
    assert likelihood_c_w_decoupled[2] == pytest.approx(0.29559074044117545)
    assert likelihood_c_w_decoupled[3] == pytest.approx(0.14415756773889613)

    likelihood_c_wo_decoupled = res["likelihood_c_wo_decoupled"]

    assert likelihood_c_wo_decoupled[0] == pytest.approx(0.27049136885975406)
    assert likelihood_c_wo_decoupled[1] == pytest.approx(0.18069487505980106)
    assert likelihood_c_wo_decoupled[2] == pytest.approx(0.2946312567418658)
    assert likelihood_c_wo_decoupled[3] == pytest.approx(0.1756221148697725)

    LRT_c = res["LRT_c"]

    assert LRT_c[0] == pytest.approx(412.6377302581877)
    assert LRT_c[1] == pytest.approx(387.17768784670744)
    assert LRT_c[2] == pytest.approx(41.85030417680922)
    assert LRT_c[3] == pytest.approx(-2541.2892308686864)

    pval_c = res["pval_c"]

    assert pval_c[0] == pytest.approx(9.77157958123275e-92)
    assert pval_c[1] == pytest.approx(3.40646307545716e-86)
    assert pval_c[2] == pytest.approx(9.853544236389633e-11)
    assert pval_c[3] == pytest.approx(1.000000e+00)

    likelihood_w_decoupled = res["likelihood_w_decoupled"]

    assert likelihood_w_decoupled[0] == pytest.approx(0.17797855757970318)
    assert likelihood_w_decoupled[1] == pytest.approx(0.008452567914799253)
    assert likelihood_w_decoupled[2] == pytest.approx(0.1401558170496066)
    assert likelihood_w_decoupled[3] == pytest.approx(0.005028937156769461)

    likelihood_wo_decoupled = res["likelihood_wo_decoupled"]

    assert likelihood_wo_decoupled[0] == pytest.approx(0.18131709692361747)
    assert likelihood_wo_decoupled[1] == pytest.approx(0.009486336541048204)
    assert likelihood_wo_decoupled[2] == pytest.approx(0.1413673401946781)
    assert likelihood_wo_decoupled[3] == pytest.approx(0.008298790494027474)

    LRT = res["LRT"]

    assert LRT[0] == pytest.approx(-239.21756210681016)
    assert LRT[1] == pytest.approx(-1485.199859092484)
    assert LRT[2] == pytest.approx(-110.78891233841415)
    assert LRT[3] == pytest.approx(-6447.5992118679505)

    pval = res["pval"]

    assert pval[0] == pytest.approx(1.0)
    assert pval[1] == pytest.approx(1.0)
    assert pval[2] == pytest.approx(1.0)
    assert pval[3] == pytest.approx(1.0)

    c_likelihood = res["likelihood_c_w_decoupled"]

    assert c_likelihood[0] == pytest.approx(0.27930299376829615)
    assert c_likelihood[1] == pytest.approx(0.1862125750499589)
    assert c_likelihood[2] == pytest.approx(0.29559074044117545)
    assert c_likelihood[3] == pytest.approx(0.14415756773889613)

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


def test_wnn():

    # read in the original AnnData objects
    
    print("Reading!")
    wnn_rna = sc.read(rna_path)
    wnn_atac = sc.read(atac_path)

    print("**********************************")
    print(wnn_rna)
    print()
    print(wnn_atac)
    print("**********************************")

    sc.pp.highly_variable_genes(wnn_rna, n_top_genes=2000)
    nn_idx, nn_dist = a.gen_wnn(wnn_rna, wnn_atac, dims=[50, 50], nn=20)

    assert nn_idx.shape[0] == 6436
    assert nn_idx.shape[1] == 20

    assert nn_dist.shape[0] == 6436
    assert nn_dist.shape[1] == 20

    height_midpoint = math.floor(nn_idx.shape[0] / 2)
    width_midpoint = math.floor(nn_idx.shape[1] / 2)

    height_endpoint = nn_idx.shape[0] - 1
    width_endpoint = nn_idx.shape[1] - 1

    height_coords = [0, height_midpoint, height_endpoint]
    width_coords = [0, width_midpoint, width_endpoint]

    idx_true = [418.0, 2681.0, 5399.0, 3082.0, 3521.0, 4206.0, 5508.0, 5890.0, 6366.0]

    dist_true = [0.3655691795516853, 0.42251151972196666, 0.42925116265230784, 0.3950307537119378, 0.37748548476070853, 0.3443918113878544, 0.37667416881716814, 0.3358022105217967, 0.3144874637270965]

    i = 0

    for h in height_coords:
        for w in width_coords:

            assert nn_idx[h][w] == idx_true[i]
            assert nn_dist[h][w] == pytest.approx(dist_true[i], rel=1e-3)

            i += 1
