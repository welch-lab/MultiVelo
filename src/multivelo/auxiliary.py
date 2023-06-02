import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, diags
from umap.umap_ import fuzzy_simplicial_set
from anndata import AnnData
import scanpy as sc
import scvelo as scv
import pandas as pd
from tqdm.auto import tqdm
import scipy
import sys
import os

from .pyWNN import *


def aggregate_peaks_10x(adata_atac, peak_annot_file, linkage_file,
                        peak_dist=10000, min_corr=0.5, gene_body=False,
                        return_dict=False, verbose=False):

    """Peak to gene aggregation.

    This function aggregates promoter and enhancer peaks to genes based on the
    10X linkage file.

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object which stores raw peak counts.
    peak_annot_file: `str`
        Peak annotation file from 10X CellRanger ARC.
    linkage_file: `str`
        Peak-gene linkage file from 10X CellRanger ARC. This file stores highly
        correlated peak-peak and peak-gene pair information.
    peak_dist: `int` (default: 10000)
        Maximum distance for peaks to be included for a gene.
    min_corr: `float` (default: 0.5)
        Minimum correlation for a peak to be considered as enhancer.
    gene_body: `bool` (default: `False`)
        Whether to add gene body peaks to the associated promoters.
    return_dict: `bool` (default: `False`)
        Whether to return promoter and enhancer dictionaries.
    verbose: `bool` (default: `False`)
        Whether to print number of genes with promoter peaks.

    Returns
    -------
    A new ATAC anndata object which stores gene aggreagted peak counts.
    Additionally, if `return_dict==True`:
        A dictionary which stores genes and promoter peaks.
        And a dictionary which stores genes and enhancer peaks.
    """
    promoter_dict = {}
    distal_dict = {}
    gene_body_dict = {}
    corr_dict = {}

    # read annotations
    with open(peak_annot_file) as f:
        header = next(f)
        tmp = header.split('\t')
        if len(tmp) == 4:
            cellranger_version = 1
        elif len(tmp) == 6:
            cellranger_version = 2
        else:
            raise ValueError('Peak annotation file should contain 4 columns '
                             '(CellRanger ARC 1.0.0) or 5 columns (CellRanger '
                             'ARC 2.0.0)')
        if verbose:
            print(f'CellRanger ARC identified as {cellranger_version}.0.0')
        if cellranger_version == 1:
            for line in f:
                tmp = line.rstrip().split('\t')
                tmp1 = tmp[0].split('_')
                peak = f'{tmp1[0]}:{tmp1[1]}-{tmp1[2]}'
                if tmp[1] != '':
                    genes = tmp[1].split(';')
                    dists = tmp[2].split(';')
                    types = tmp[3].split(';')
                    for i, gene in enumerate(genes):
                        dist = dists[i]
                        annot = types[i]
                        if annot == 'promoter':
                            if gene not in promoter_dict:
                                promoter_dict[gene] = [peak]
                            else:
                                promoter_dict[gene].append(peak)
                        elif annot == 'distal':
                            if dist == '0':
                                if gene not in gene_body_dict:
                                    gene_body_dict[gene] = [peak]
                                else:
                                    gene_body_dict[gene].append(peak)
                            else:
                                if gene not in distal_dict:
                                    distal_dict[gene] = [peak]
                                else:
                                    distal_dict[gene].append(peak)
        else:
            for line in f:
                tmp = line.rstrip().split('\t')
                peak = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                gene = tmp[3]
                dist = tmp[4]
                annot = tmp[5]
                if annot == 'promoter':
                    if gene not in promoter_dict:
                        promoter_dict[gene] = [peak]
                    else:
                        promoter_dict[gene].append(peak)
                elif annot == 'distal':
                    if dist == '0':
                        if gene not in gene_body_dict:
                            gene_body_dict[gene] = [peak]
                        else:
                            gene_body_dict[gene].append(peak)
                    else:
                        if gene not in distal_dict:
                            distal_dict[gene] = [peak]
                        else:
                            distal_dict[gene].append(peak)

    # read linkages
    with open(linkage_file) as f:
        for line in f:
            tmp = line.rstrip().split('\t')
            if tmp[12] == "peak-peak":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    for t3 in tmp3:
                        gene2 = t3.split('_')
                        # one of the peaks is in promoter, peaks belong to the
                        # same gene or are close in distance
                        if (((gene1[1] == "promoter") !=
                            (gene2[1] == "promoter")) and
                            ((gene1[0] == gene2[0]) or
                             (float(tmp[11]) < peak_dist))):

                            if gene1[1] == "promoter":
                                gene = gene1[0]
                            else:
                                gene = gene2[0]
                            if gene in corr_dict:
                                # peak 1 is in promoter, peak 2 is not in gene
                                # body -> peak 2 is added to gene 1
                                if (peak2 not in corr_dict[gene] and
                                    gene1[1] == "promoter" and
                                    (gene2[0] not in gene_body_dict or
                                     peak2 not in gene_body_dict[gene2[0]])):

                                    corr_dict[gene][0].append(peak2)
                                    corr_dict[gene][1].append(corr)
                                # peak 2 is in promoter, peak 1 is not in gene
                                # body -> peak 1 is added to gene 2
                                if (peak1 not in corr_dict[gene] and
                                    gene2[1] == "promoter" and
                                    (gene1[0] not in gene_body_dict or
                                     peak1 not in gene_body_dict[gene1[0]])):

                                    corr_dict[gene][0].append(peak1)
                                    corr_dict[gene][1].append(corr)
                            else:
                                # peak 1 is in promoter, peak 2 is not in gene
                                # body -> peak 2 is added to gene 1
                                if (gene1[1] == "promoter" and
                                    (gene2[0] not in
                                     gene_body_dict
                                     or peak2 not in
                                     gene_body_dict[gene2[0]])):

                                    corr_dict[gene] = [[peak2], [corr]]
                                # peak 2 is in promoter, peak 1 is not in gene
                                # body -> peak 1 is added to gene 2
                                if (gene2[1] == "promoter" and
                                    (gene1[0] not in
                                     gene_body_dict
                                     or peak1 not in
                                     gene_body_dict[gene1[0]])):

                                    corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "peak-gene":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                gene2 = tmp[6].split('><')[1][:-1]
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    # peak 1 belongs to gene 2 or are close in distance
                    # -> peak 1 is added to gene 2
                    if ((gene1[0] == gene2) or (float(tmp[11]) < peak_dist)):
                        gene = gene1[0]
                        if gene in corr_dict:
                            if (peak1 not in corr_dict[gene] and
                                gene1[1] != "promoter" and
                                (gene1[0] not in gene_body_dict or
                                 peak1 not in gene_body_dict[gene1[0]])):

                                corr_dict[gene][0].append(peak1)
                                corr_dict[gene][1].append(corr)
                        else:
                            if (gene1[1] != "promoter" and
                                (gene1[0] not in gene_body_dict or
                                 peak1 not in gene_body_dict[gene1[0]])):
                                corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "gene-peak":
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                gene1 = tmp[6].split('><')[0][1:]
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t3 in tmp3:
                    gene2 = t3.split('_')
                    # peak 2 belongs to gene 1 or are close in distance
                    # -> peak 2 is added to gene 1
                    if ((gene1 == gene2[0]) or (float(tmp[11]) < peak_dist)):
                        gene = gene1
                        if gene in corr_dict:
                            if (peak2 not in corr_dict[gene] and
                                gene2[1] != "promoter" and
                                (gene2[0] not in gene_body_dict or
                                 peak2 not in gene_body_dict[gene2[0]])):

                                corr_dict[gene][0].append(peak2)
                                corr_dict[gene][1].append(corr)
                        else:
                            if (gene2[1] != "promoter" and
                                (gene2[0] not in gene_body_dict or
                                 peak2 not in gene_body_dict[gene2[0]])):

                                corr_dict[gene] = [[peak2], [corr]]

    gene_dict = promoter_dict
    enhancer_dict = {}
    promoter_genes = list(promoter_dict.keys())
    if verbose:
        print(f'Found {len(promoter_genes)} genes with promoter peaks')
    for gene in promoter_genes:
        if gene_body:  # add gene-body peaks
            if gene in gene_body_dict:
                for peak in gene_body_dict[gene]:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
        enhancer_dict[gene] = []
        if gene in corr_dict:  # add enhancer peaks
            for j, peak in enumerate(corr_dict[gene][0]):
                corr = corr_dict[gene][1][j]
                if corr > min_corr:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
                        enhancer_dict[gene].append(peak)

    # aggregate to genes
    adata_atac_X_copy = adata_atac.X.A
    gene_mat = np.zeros((adata_atac.shape[0], len(promoter_genes)))
    var_names = adata_atac.var_names.to_numpy()
    for i, gene in tqdm(enumerate(promoter_genes), total=len(promoter_genes)):
        peaks = gene_dict[gene]
        for peak in peaks:
            if peak in var_names:
                peak_index = np.where(var_names == peak)[0][0]
                gene_mat[:, i] += adata_atac_X_copy[:, peak_index]
    gene_mat[gene_mat < 0] = 0
    gene_mat = AnnData(X=csr_matrix(gene_mat))
    gene_mat.obs_names = pd.Index(list(adata_atac.obs_names))
    gene_mat.var_names = pd.Index(promoter_genes)
    gene_mat = gene_mat[:, gene_mat.X.sum(0) > 0]
    if return_dict:
        return gene_mat, promoter_dict, enhancer_dict
    else:
        return gene_mat


def tfidf_norm(adata_atac, scale_factor=1e4, copy=False):
    """TF-IDF normalization.

    This function normalizes counts in an AnnData object with TF-IDF.

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    scale_factor: `float` (default: 1e4)
        Value to be multiplied after normalization.
    copy: `bool` (default: `False`)
        Whether to return a copy or modify `.X` directly.

    Returns
    -------
    If `copy==True`, a new ATAC anndata object which stores normalized counts
    in `.X`.
    """
    npeaks = adata_atac.X.sum(1)
    npeaks_inv = csr_matrix(1.0/npeaks)
    tf = adata_atac.X.multiply(npeaks_inv)
    idf = diags(np.ravel(adata_atac.X.shape[0] / adata_atac.X.sum(0))).log1p()
    if copy:
        adata_atac_copy = adata_atac.copy()
        adata_atac_copy.X = tf.dot(idf) * scale_factor
        return adata_atac_copy
    else:
        adata_atac.X = tf.dot(idf) * scale_factor


def gen_wnn(adata_rna, adata_adt, dims, nn):
    """Inputs for KNN smoothing.

    This function calculates the nn_idx and nn_dist matrices needed
    to run knn_smooth_chrom().

    Parameters
    ----------
    adata_rna: :class:`~anndata.AnnData`
        RNA anndata object.
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    dims: `List[int]`
        Dimensions of data for RNA (index=0) and ATAC (index=1)
    nn: `int` (default: `None`)
        Top N neighbors to extract for each cell in the connectivities matrix.

    Returns
    -------
    nn_idx: `np.darray` (default: `None`)
        KNN index matrix of size (cells, k).
    nn_dist: `np.darray` (default: `None`)
        KNN distance matrix of size (cells, k).
    """

    # make a copy of the original adata objects so as to keep them unchanged
    rna_copy = adata_rna.copy()
    adt_copy = adata_adt.copy()

    sc.tl.pca(rna_copy,
              n_comps=dims[0],
              random_state=np.random.RandomState(seed=42),
              use_highly_variable=True)  # run PCA on RNA

    lsi = scipy.sparse.linalg.svds(adt_copy.X, k=dims[1])  # run SVD on ADT

    # get the lsi result
    adt_copy.obsm['X_lsi'] = lsi[0]

    # add the PCA from adt to rna
    rna_copy.obsm['X_rna_pca'] = rna_copy.obsm.pop('X_pca')
    rna_copy.obsm['X_adt_lsi'] = adt_copy.obsm['X_lsi']

    # run WNN
    WNNobj = pyWNN(rna_copy,
                   reps=['X_rna_pca', 'X_adt_lsi'],
                   npcs=dims,
                   n_neighbors=nn,
                   seed=42)

    adata_seurat = WNNobj.compute_wnn(rna_copy)

    # get the matrix storing the distances between each cell and its neighbors
    cx = scipy.sparse.coo_matrix(adata_seurat.obsp["WNN_distance"])

    # the number of cells
    cells = adata_seurat.obsp['WNN_distance'].shape[0]

    # define the shape of our final results
    # and make the arrays that will hold the results
    new_shape = (cells, nn)
    nn_dist = np.zeros(shape=new_shape)
    nn_idx = np.zeros(shape=new_shape)

    # new_col defines what column we store data in
    # our result arrays
    new_col = 0

    # loop through the distance matrices
    for i, j, v in zip(cx.row, cx.col, cx.data):

        # store the distances between neighbor cells
        nn_dist[i][new_col % nn] = v

        # for each cell's row, store the row numbers of its neighbor cells
        # (1-indexing instead of 0- is a holdover from R multimodalneighbors())
        nn_idx[i][new_col % nn] = int(j) + 1

        new_col += 1

    return nn_idx, nn_dist


def knn_smooth_chrom(adata_atac, nn_idx=None, nn_dist=None, conn=None,
                     n_neighbors=None):
    """KNN smoothing.

    This function smooth (impute) the count matrix with k nearest neighbors.
    The inputs can be either KNN index and distance matrices or a pre-computed
    connectivities matrix (for example in adata_rna object).

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object.
    nn_idx: `np.darray` (default: `None`)
        KNN index matrix of size (cells, k).
    nn_dist: `np.darray` (default: `None`)
        KNN distance matrix of size (cells, k).
    conn: `csr_matrix` (default: `None`)
        Pre-computed connectivities matrix.
    n_neighbors: `int` (default: `None`)
        Top N neighbors to extract for each cell in the connectivities matrix.

    Returns
    -------
    `.layers['Mc']` stores imputed values.
    """
    if nn_idx is not None and nn_dist is not None:
        if nn_idx.shape[0] != adata_atac.shape[0]:
            raise ValueError('Number of rows of KNN indices does not equal to '
                             'number of observations.')
        if nn_dist.shape[0] != adata_atac.shape[0]:
            raise ValueError('Number of rows of KNN distances does not equal '
                             'to number of observations.')
        X = coo_matrix(([], ([], [])), shape=(nn_idx.shape[0], 1))
        conn, sigma, rho, dists = fuzzy_simplicial_set(X, nn_idx.shape[1],
                                                       None, None,
                                                       knn_indices=nn_idx-1,
                                                       knn_dists=nn_dist,
                                                       return_dists=True)
    elif conn is not None:
        pass
    else:
        raise ValueError('Please input nearest neighbor indices and distances,'
                         ' or a connectivities matrix of size n x n, with '
                         'columns being neighbors.'
                         ' For example, RNA connectivities can usually be '
                         'found in adata.obsp.')

    conn = conn.tocsr().copy()
    n_counts = (conn > 0).sum(1).A1
    if n_neighbors is not None and n_neighbors < n_counts.min():
        conn = top_n_sparse(conn, n_neighbors)
    conn.setdiag(1)
    conn_norm = conn.multiply(1.0 / conn.sum(1)).tocsr()
    adata_atac.layers['Mc'] = csr_matrix.dot(conn_norm, adata_atac.X)
    adata_atac.obsp['connectivities'] = conn


def calculate_qc_metrics(adata, **kwargs):
    """Basic QC metrics.

    This function calculate basic QC metrics with
    `scanpy.pp.calculate_qc_metrics`.
    Additionally, total counts and the ratio of unspliced and spliced matrices,
    as well as the cell cycle scores (with `scvelo.tl.score_genes_cell_cycle`)
    will be computed.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        RNA anndata object. Required fields: `unspliced` and `spliced`.
    Additional parameters passed to `scanpy.pp.calculate_qc_metrics`.

    Returns
    -------
    Outputs of `scanpy.pp.calculate_qc_metrics` and
    `scvelo.tl.score_genes_cell_cycle`. total_unspliced, total_spliced: `.var`
        total counts of unspliced and spliced matrices.
    unspliced_ratio: `.var`
        ratio of unspliced counts vs (unspliced + spliced counts).
    cell_cycle_score: `.var`
        cell cycle score difference between G2M_score and S_score.
    """
    print("Running from here!")
    sc.pp.calculate_qc_metrics(adata, **kwargs)
    if 'spliced' not in adata.layers:
        raise ValueError('Spliced matrix not found in adata.layers')
    if 'unspliced' not in adata.layers:
        raise ValueError('Unspliced matrix not found in adata.layers')
    print(adata.layers['spliced'].shape)
    total_s = np.nansum(adata.layers['spliced'].toarray(), axis=1)
    total_u = np.nansum(adata.layers['unspliced'].toarray(), axis=1)
    print(total_u.shape)
    adata.obs['total_unspliced'] = total_u
    adata.obs['total_spliced'] = total_s
    adata.obs['unspliced_ratio'] = total_u / (total_s + total_u)
    scv.tl.score_genes_cell_cycle(adata)
    adata.obs['cell_cycle_score'] = (adata.obs['G2M_score']
                                     - adata.obs['S_score'])


def ellipse_fit(adata,
                genes,
                color_by='quantile',
                n_cols=8,
                title=None,
                figsize=None,
                axis_on=False,
                pointsize=2,
                linewidth=2
                ):
    """Fit ellipses to unspliced and spliced phase portraits.

    This function plots the ellipse fits on the unspliced-spliced phase
    portraits.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        RNA anndata object. Required fields: `Mu` and `Ms`.
    genes: `str`,  list of `str`
        List of genes to plot.
    color_by: `str` (default: `quantile`)
        Color by the four quantiles based on ellipse fit if `quantile`. Other
        common values are leiden, louvain, celltype, etc.
        If not `quantile`, the color field must be present in `.uns`, which
        can be pre-computed with `scanpy.pl.scatter`.
        For `quantile`, red, orange, green, and blue represent quantile left,
        top, right, and bottom, respectively.
        If `quantile_scores`, `multivelo.compute_quantile_scores` function
        must have been run.
    n_cols: `int` (default: 8)
        Number of columns to plot on each row.
    figsize: `tuple` (default: `None`)
        Total figure size.
    title: `tuple` (default: `None`)
        Title of the figure. Default is `Ellipse Fit`.
    axis_on: `bool` (default: `False`)
        Whether to show axis labels.
    pointsize: `float` (default: 2)
        Point size for scatter plots.
    linewidth: `float` (default: 2)
        Line width for ellipse.
    """
    by_quantile = color_by == 'quantile'
    by_quantile_score = color_by == 'quantile_scores'
    if not by_quantile and not by_quantile_score:
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    gn = len(genes)
    if gn < n_cols:
        n_cols = gn
    fig, axs = plt.subplots(-(-gn // n_cols), n_cols, figsize=(2 * n_cols,
                            2.4 * (-(-gn // n_cols)))
                            if figsize is None else figsize)
    count = 0
    for gene in genes:
        u = np.array(adata[:, gene].layers['Mu'])
        s = np.array(adata[:, gene].layers['Ms'])
        row = count // n_cols
        col = count % n_cols
        non_zero = (u > 0) & (s > 0)
        if np.sum(non_zero) < 10:
            count += 1
            fig.delaxes(axs[row, col])
            continue

        mean_u, mean_s = np.mean(u[non_zero]), np.mean(s[non_zero])
        std_u, std_s = np.std(u[non_zero]), np.std(s[non_zero])
        u_ = (u - mean_u)/std_u
        s_ = (s - mean_s)/std_s
        X = np.reshape(s_[non_zero], (-1, 1))
        Y = np.reshape(u_[non_zero], (-1, 1))

        # Ax^2 + Bxy + Cy^2 + Dx + Ey + 1 = 0
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = -np.ones_like(X)
        x, res, _, _ = np.linalg.lstsq(A, b)
        x = x.squeeze()
        A, B, C, D, E = x
        good_fit = B**2 - 4*A*C < 0
        theta = np.arctan(B/(A - C))/2 \
            if x[0] > x[2] \
            else np.pi/2 + np.arctan(B/(A - C))/2
        good_fit = good_fit & (theta < np.pi/2) & (theta > 0)
        if not good_fit:
            count += 1
            fig.delaxes(axs[row, col])
            continue
        x_coord = np.linspace((-mean_s)/std_s, (np.max(s)-mean_s)/std_s, 500)
        y_coord = np.linspace((-mean_u)/std_u, (np.max(u)-mean_u)/std_u, 500)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = (A * X_coord**2 + B * X_coord * Y_coord + C * Y_coord**2 +
                   D * X_coord + E * Y_coord + 1)

        M0 = np.array([
             A, B/2, D/2,
             B/2, C, E/2,
             D/2, E/2, 1,
        ]).reshape(3, 3)
        M = np.array([
            A, B/2,
            B/2, C,
        ]).reshape(2, 2)
        l1, l2 = np.sort(np.linalg.eigvals(M))
        xc = (B*E - 2*C*D)/(4*A*C - B**2)
        yc = (B*D - 2*A*E)/(4*A*C - B**2)
        slope_major = np.tan(theta)
        theta2 = np.pi/2 + theta
        slope_minor = np.tan(theta2)
        a = np.sqrt(-np.linalg.det(M0)/np.linalg.det(M)/l2)
        b = np.sqrt(-np.linalg.det(M0)/np.linalg.det(M)/l1)
        xtop = xc + a*np.cos(theta)
        ytop = yc + a*np.sin(theta)
        xbot = xc - a*np.cos(theta)
        ybot = yc - a*np.sin(theta)
        xtop2 = xc + b*np.cos(theta2)
        ytop2 = yc + b*np.sin(theta2)
        xbot2 = xc - b*np.cos(theta2)
        ybot2 = yc - b*np.sin(theta2)
        mse = res[0] / np.sum(non_zero)
        major = lambda x, y: (y - yc) - (slope_major * (x - xc))
        minor = lambda x, y: (y - yc) - (slope_minor * (x - xc))
        quant1 = (major(s_, u_) > 0) & (minor(s_, u_) < 0)
        quant2 = (major(s_, u_) > 0) & (minor(s_, u_) > 0)
        quant3 = (major(s_, u_) < 0) & (minor(s_, u_) > 0)
        quant4 = (major(s_, u_) < 0) & (minor(s_, u_) < 0)
        if (np.sum(quant1 | quant4) < 10) or (np.sum(quant2 | quant3) < 10):
            count += 1
            continue

        if by_quantile:
            axs[row, col].scatter(s_[quant1], u_[quant1], s=pointsize,
                                  c='tab:red', alpha=0.6)
            axs[row, col].scatter(s_[quant2], u_[quant2], s=pointsize,
                                  c='tab:orange', alpha=0.6)
            axs[row, col].scatter(s_[quant3], u_[quant3], s=pointsize,
                                  c='tab:green', alpha=0.6)
            axs[row, col].scatter(s_[quant4], u_[quant4], s=pointsize,
                                  c='tab:blue', alpha=0.6)
        elif by_quantile_score:
            if 'quantile_scores' not in adata.layers:
                raise ValueError('Please run multivelo.compute_quantile_scores'
                                 ' first to compute quantile scores.')
            axs[row, col].scatter(s_, u_, s=pointsize,
                                  c=adata[:, gene].layers['quantile_scores'],
                                  cmap='RdBu_r', alpha=0.7)
        else:
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                axs[row, col].scatter(s_[filt], u_[filt], s=pointsize,
                                      c=colors[i], alpha=0.7)
        axs[row, col].contour(X_coord, Y_coord, Z_coord, levels=[0],
                              colors=('r'), linewidths=linewidth, alpha=0.7)
        axs[row, col].scatter([xc], [yc], c='black', s=5, zorder=2)
        axs[row, col].scatter([0], [0], c='black', s=5, zorder=2)
        axs[row, col].plot([xtop, xbot], [ytop, ybot], color='b',
                           linestyle='dashed', linewidth=linewidth, alpha=0.7)
        axs[row, col].plot([xtop2, xbot2], [ytop2, ybot2], color='g',
                           linestyle='dashed', linewidth=linewidth, alpha=0.7)

        axs[row, col].set_title(f'{gene} {mse:.3g}')
        axs[row, col].set_xlabel('s')
        axs[row, col].set_ylabel('u')
        common_range = [(np.min([(-mean_s)/std_s, (-mean_u)/std_u])
                        - (0.05*np.max(s)/std_s)),
                        (np.max([(np.max(s)-mean_s)/std_s,
                                 (np.max(u)-mean_u)/std_u])
                        + (0.05*np.max(s)/std_s))]
        axs[row, col].set_xlim(common_range)
        axs[row, col].set_ylim(common_range)
        if not axis_on:
            axs[row, col].xaxis.set_ticks_position('none')
            axs[row, col].yaxis.set_ticks_position('none')
            axs[row, col].get_xaxis().set_visible(False)
            axs[row, col].get_yaxis().set_visible(False)
            axs[row, col].xaxis.set_ticks_position('none')
            axs[row, col].yaxis.set_ticks_position('none')
            axs[row, col].set_frame_on(False)
        count += 1

    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    if title is not None:
        fig.suptitle(title, fontsize=15)
    else:
        fig.suptitle('Ellipse Fit', fontsize=15)
    fig.tight_layout(rect=[0, 0.1, 1, 0.98])


def compute_quantile_scores(adata,
                            verbose=True,
                            n_pcs=30,
                            n_neighbors=30
                            ):
    """Fit ellipses to unspliced and spliced phase portraits and compute
        quantile scores.

    This function fit ellipses to unspliced-spliced phase portraits. The cells
    are split into four groups (quantiles) based on the axes of the ellipse.
    Then the function assigns each quantile a score: -3 for left, -1 for top, 1
    for right, and 3 for bottom. These gene-specific values are smoothed with a
    connectivities matrix. This is similar to the RNA velocity gene time
    assignment.

    In addition, a 2-bit tuple is assigned to each of the four quantiles, (0,0)
    for left, (1,0) for top, (1,1) for right, and (0,1) for bottom. This is to
    mimic the distance relationship between quantiles.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        RNA anndata object. Required fields: `Mu` and `Ms`.
    verbose: `bool` (default: `True`)
        Print the number of good ellipse fit based on some criteria.
    n_pcs: `int` (default: 30)
        Number of principal components to compute connectivities.
    n_neighbors: `int` (default: 30)
        Number of nearest neighbors to compute connectivities.

    Returns
    -------
    quantile_scores: `.layers`
        gene-specific quantile scores
    quantile_scores_1st_bit, quantile_scores_2nd_bit: `.layers`
        2-bit assignment for gene quantiles
    quantile_score_sum: `.obs`
        aggreagted quantile scores
    quantile_genes: `.var`
        genes with good quantilty ellipse fits
    """
    neighbors = Neighbors(adata)
    neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, n_pcs=n_pcs)
    conn = neighbors.connectivities
    conn.setdiag(1)
    conn_norm = conn.multiply(1.0 / conn.sum(1)).tocsr()

    quantile_scores = np.zeros(adata.shape)
    quantile_scores_2bit = np.zeros((adata.shape[0], adata.shape[1], 2))
    quantile_gene = np.full(adata.n_vars, False)
    quality_gene_idx = []
    for idx, gene in enumerate(adata.var_names):
        u = np.array(adata[:, gene].layers['Mu'])
        s = np.array(adata[:, gene].layers['Ms'])
        non_zero = (u > 0) & (s > 0)
        if np.sum(non_zero) < 10:
            continue

        mean_u, mean_s = np.mean(u[non_zero]), np.mean(s[non_zero])
        std_u, std_s = np.std(u[non_zero]), np.std(s[non_zero])
        u_ = (u - mean_u)/std_u
        s_ = (s - mean_s)/std_s
        X = np.reshape(s_[non_zero], (-1, 1))
        Y = np.reshape(u_[non_zero], (-1, 1))

        # Ax^2 + Bxy + Cy^2 + Dx + Ey + 1 = 0
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = -np.ones_like(X)
        x, res, _, _ = np.linalg.lstsq(A, b)
        x = x.squeeze()
        A, B, C, D, E = x
        good_fit = B**2 - 4*A*C < 0
        theta = np.arctan(B/(A - C))/2 \
            if x[0] > x[2] \
            else np.pi/2 + np.arctan(B/(A - C))/2
        good_fit = good_fit & (theta < np.pi/2) & (theta > 0)
        if not good_fit:
            continue

        x_coord = np.linspace((-mean_s)/std_s, (np.max(s)-mean_s)/std_s, 500)
        y_coord = np.linspace((-mean_u)/std_u, (np.max(u)-mean_u)/std_u, 500)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        M = np.array([
            A, B/2,
            B/2, C,
        ]).reshape(2, 2)
        l1, l2 = np.sort(np.linalg.eigvals(M))
        xc = (B*E - 2*C*D)/(4*A*C - B**2)
        yc = (B*D - 2*A*E)/(4*A*C - B**2)
        slope_major = np.tan(theta)
        theta2 = np.pi/2 + theta
        slope_minor = np.tan(theta2)
        major = lambda x, y: (y - yc) - (slope_major * (x - xc))
        minor = lambda x, y: (y - yc) - (slope_minor * (x - xc))

        quant1 = (major(s_, u_) > 0) & (minor(s_, u_) < 0)
        quant2 = (major(s_, u_) > 0) & (minor(s_, u_) > 0)
        quant3 = (major(s_, u_) < 0) & (minor(s_, u_) > 0)
        quant4 = (major(s_, u_) < 0) & (minor(s_, u_) < 0)
        if (np.sum(quant1 | quant4) < 10) or (np.sum(quant2 | quant3) < 10):
            continue

        quantile_scores[:, idx:idx+1] = ((-3.) * quant1 + (-1.) * quant2 + 1.
                                         * quant3 + 3. * quant4)
        quantile_scores_2bit[:, idx:idx+1, 0] = 1. * (quant1 | quant2)
        quantile_scores_2bit[:, idx:idx+1, 1] = 1. * (quant2 | quant3)
        quality_gene_idx.append(idx)

    quantile_scores = csr_matrix.dot(conn_norm, quantile_scores)
    quantile_scores_2bit[:, :, 0] = csr_matrix.dot(conn_norm,
                                                   quantile_scores_2bit[:,
                                                                        :, 0])
    quantile_scores_2bit[:, :, 1] = csr_matrix.dot(conn_norm,
                                                   quantile_scores_2bit[:,
                                                                        :, 1])
    adata.layers['quantile_scores'] = quantile_scores
    adata.layers['quantile_scores_1st_bit'] = quantile_scores_2bit[:, :, 0]
    adata.layers['quantile_scores_2nd_bit'] = quantile_scores_2bit[:, :, 1]
    quantile_gene[quality_gene_idx] = True

    if verbose:
        perc_good = np.sum(quantile_gene) / adata.n_vars * 100
        print(f'{np.sum(quantile_gene)}/{adata.n_vars} - {perc_good:.3g}%'
              'genes have good ellipse fits')

    adata.obs['quantile_score_sum'] = \
        np.sum(adata[:, quantile_gene].layers['quantile_scores'], axis=1)
    adata.var['quantile_genes'] = quantile_gene


def cluster_by_quantile(adata,
                        plot=False,
                        n_clusters=None,
                        affinity='euclidean',
                        linkage='ward'
                        ):
    """Cluster genes based on 2-bit quantile scores.

    This function cluster similar genes based on their 2-bit quantile score
    assignments from ellipse fit.
    Hierarchical cluster is done with `sklean.cluster.AgglomerativeClustering`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        RNA anndata object. Required fields: `Mu` and `Ms`.
    plot: `bool` (default: `False`)
        Plot the hierarchical clusters.
    n_clusters: `int` (default: None)
        The number of clusters to keep.
    affinity: `str` (default: `euclidean`)
        Metric used to compute linkage. Passed to
        `sklean.cluster.AgglomerativeClustering`.
    linkage: `str` (default: `ward`)
        Linkage criterion to use. Passed to
        `sklean.cluster.AgglomerativeClustering`.

    Returns
    -------
    quantile_cluster: `.var`
        cluster assignments of genes based on quantiles
    """
    from sklearn.cluster import AgglomerativeClustering
    if 'quantile_scores_1st_bit' not in adata.layers.keys():
        raise ValueError("Quantile scores not found. Please run "
                         "compute_quantile_scores function first.")
    quantile_gene = adata.var['quantile_genes']
    if plot or n_clusters is None:
        cluster = AgglomerativeClustering(distance_threshold=0,
                                          n_clusters=None,
                                          affinity=affinity,
                                          linkage=linkage)
        cluster = cluster.fit(np.vstack((adata[:, quantile_gene]
                                         .layers['quantile_scores_1st_bit'],
                                         adata[:, quantile_gene]
                                         .layers['quantile_scores_2nd_bit']))
                                .transpose())

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        def plot_dendrogram(model, **kwargs):
            from scipy.cluster.hierarchy import dendrogram
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count
            linkage_matrix = np.column_stack([model.children_,
                                              model.distances_,
                                              counts]).astype(float)
            dendrogram(linkage_matrix, **kwargs)

        plot_dendrogram(cluster, truncate_mode='level', p=5, no_labels=True)

    if n_clusters is not None:
        n_clusters = int(n_clusters)
        cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                          affinity=affinity,
                                          linkage=linkage)
        cluster = cluster.fit_predict(np.vstack((adata[:, quantile_gene].layers
                                                 ['quantile_scores_1st_bit'],
                                                 adata[:, quantile_gene].layers
                                                 ['quantile_scores_2nd_bit']))
                                        .transpose())
        quantile_cluster = np.full(adata.n_vars, -1)
        quantile_cluster[quantile_gene] = cluster
        adata.var['quantile_cluster'] = quantile_cluster
