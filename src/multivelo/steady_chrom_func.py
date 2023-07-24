import os
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from scanpy import Neighbors
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from multivelo import mv_logging as logg
from multivelo import settings


class ChromatinVelocity:
    def __init__(self, c, u, s,
                 ss, us,
                 gene=None,
                 save_plot=False,
                 plot_dir=None,
                 fit_args=None,
                 rna_only=False,
                 extra_color=None,
                 r2_adjusted=True,
                 ):

        self.gene = gene

        # fitting arguments
        self.rna_only = rna_only
        self.outlier = np.clip(fit_args['outlier'], 80, 100)
        self.r2_adjusted = r2_adjusted

        # plot parameters
        self.save_plot = save_plot
        self.extra_color = extra_color
        self.fig_size = fit_args['fig_size']
        self.point_size = fit_args['point_size']
        if plot_dir is None:
            self.plot_path = 'plots_steady_state'
        else:
            self.plot_path = plot_dir

        # input
        self.total_n = len(u)
        if sparse.issparse(c):
            c = c.A
        if sparse.issparse(u):
            u = u.A
        if sparse.issparse(s):
            s = s.A
        if ss is not None and sparse.issparse(ss):
            ss = ss.A
        if us is not None and sparse.issparse(us):
            us = us.A
        self.c_all = np.ravel(np.array(c, dtype=np.float64))
        self.u_all = np.ravel(np.array(u, dtype=np.float64))
        self.s_all = np.ravel(np.array(s, dtype=np.float64))
        if ss is not None:
            self.ss_all = np.ravel(np.array(ss, dtype=np.float64))
        if us is not None:
            self.us_all = np.ravel(np.array(us, dtype=np.float64))

        # adjust offset
        self.offset_c, self.offset_u, self.offset_s = np.min(self.c_all), \
            np.min(self.u_all), \
            np.min(self.s_all)
        self.offset_c = 0 if self.rna_only else self.offset_c
        self.c_all -= self.offset_c
        self.u_all -= self.offset_u
        self.s_all -= self.offset_s
        # remove zero counts
        self.non_zero = np.ravel(self.c_all > 0) | np.ravel(self.u_all > 0) | \
            np.ravel(self.s_all > 0)
        # remove outliers
        self.non_outlier = np.ravel(self.c_all <=
                                    np.percentile(self.c_all, self.outlier))
        self.non_outlier &= np.ravel(self.u_all <=
                                     np.percentile(self.u_all, self.outlier))
        self.non_outlier &= np.ravel(self.s_all <=
                                     np.percentile(self.s_all, self.outlier))
        self.c = self.c_all[self.non_zero & self.non_outlier]
        self.u = self.u_all[self.non_zero & self.non_outlier]
        self.s = self.s_all[self.non_zero & self.non_outlier]
        self.ss = (None if ss is None
                   else self.ss_all[self.non_zero & self.non_outlier])
        self.us = (None if us is None
                   else self.us_all[self.non_zero & self.non_outlier])
        self.low_quality = len(self.u) < 10

        logg.update(f'{len(self.u)} cells passed filter and will be used to '
                    'fit regressions.', v=2)

        # 4 rate parameters
        self.alpha_c = 0.1
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0

        # other parameters or results
        self.loss = np.inf
        self.r2 = 0
        self.residual = None
        self.residual2 = None
        self.steady_state_func = None

        # select cells for regression
        w_sub = (self.c >= 0.1 * np.max(self.c)) & \
                (self.u >= 0.1 * np.max(self.u)) & \
                (self.s >= 0.1 * np.max(self.s))
        c_sub = self.c[w_sub]
        if not self.rna_only:
            w_sub = (self.c >= np.mean(c_sub)+np.std(c_sub)) & \
                    (self.u >= 0.1 * np.max(self.u)) & \
                    (self.s >= 0.1 * np.max(self.s))
        self.w_sub = w_sub
        if np.sum(self.w_sub) < 10:
            self.low_quality = True

    def compute_deterministic(self):
        if self.rna_only:
            # steady-state slope
            wu = self.u >= np.percentile(self.u[self.w_sub], 95)
            ws = self.s >= np.percentile(self.s[self.w_sub], 95)
            ss_u = self.u[wu | ws]
            ss_s = self.s[wu | ws]
        else:
            # chromatin adjusted steady-state slope
            u_high = self.u[self.w_sub]
            s_high = self.s[self.w_sub]
            wu_high = u_high >= np.percentile(u_high, 95)
            ws_high = s_high >= np.percentile(s_high, 95)
            ss_u = u_high[wu_high | ws_high]
            ss_s = s_high[wu_high | ws_high]
        gamma = np.dot(ss_u, ss_s) / np.dot(ss_s, ss_s)
        self.steady_state_func = lambda x: gamma*x
        residual = self.u_all - self.steady_state_func(self.s_all)
        self.residual = residual
        self.loss = np.dot(self.residual, self.residual) / len(self.u_all)

        if self.r2_adjusted:
            gamma = np.dot(self.u, self.s) / np.dot(self.s, self.s)
            residual = self.u_all - gamma * self.s_all

        total = self.u_all - np.mean(self.u_all)
        self.r2 = 1 - np.dot(residual, residual) / np.dot(total, total)

    def compute_stochastic(self):
        self.compute_deterministic()

        var_ss = 2 * self.ss - self.s
        cov_us = 2 * self.us + self.u
        s_all_ = 2 * self.s_all**2 - (2 * self.ss_all - self.s_all)
        u_all_ = (2 * self.us_all + self.u_all) - 2 * self.u_all*self.s_all
        gamma2 = np.dot(cov_us, var_ss) / np.dot(var_ss, var_ss)
        residual2 = cov_us - gamma2 * var_ss
        std_first = np.std(self.residual)
        std_second = np.std(residual2)

        # combine first and second moments and recompute gamma
        if self.rna_only:
            # steady-state slope
            wu = self.u >= np.percentile(self.u[self.w_sub], 95)
            ws = self.s >= np.percentile(self.s[self.w_sub], 95)
            ss_u = self.u * (wu | ws)
            ss_s = self.s * (wu | ws)
            a = np.hstack((ss_s / std_first, var_ss / std_second))
            b = np.hstack((ss_u / std_first, cov_us / std_second))
        else:
            # chromatin adjusted steady-state slope
            u_high = self.u[self.w_sub]
            s_high = self.s[self.w_sub]
            wu_high = u_high >= np.percentile(u_high, 95)
            ws_high = s_high >= np.percentile(s_high, 95)
            ss_u = u_high * (wu_high | ws_high)
            ss_s = s_high * (wu_high | ws_high)
            a = np.hstack((ss_s / std_first, var_ss[self.w_sub] / std_second))
            b = np.hstack((ss_u / std_first, cov_us[self.w_sub] / std_second))
        gamma = np.dot(b, a) / np.dot(a, a)
        self.steady_state_func = lambda x: gamma*x
        self.residual = self.u_all - self.steady_state_func(self.s_all)
        self.residual2 = u_all_ - self.steady_state_func(s_all_)
        self.loss = np.dot(self.residual, self.residual) / len(self.u_all)

    def get_velocity(self):
        return self.residual

    def get_variance_velocity(self):
        return self.residual2

    def get_r2(self):
        return self.r2

    def get_loss(self):
        return self.loss


def regress_func(c, u, s, ss, us, m, sp, pdir, fa, gene, ro, extra):

    c_90 = np.percentile(c, 90)
    u_90 = np.percentile(u, 90)
    s_90 = np.percentile(s, 90)
    low_quality = ((u_90 == 0 or s_90 == 0) if ro
                   else (c_90 == 0 or u_90 == 0 or s_90 == 0))
    if low_quality:
        logg.update(f'low quality gene {gene}, skipping', v=1)
        return np.zeros(len(u)), np.zeros(len(u)), 0, np.inf

    cvc = ChromatinVelocity(c,
                            u,
                            s,
                            ss,
                            us,
                            save_plot=sp,
                            plot_dir=pdir,
                            fit_args=fa,
                            gene=gene,
                            rna_only=ro,
                            extra_color=extra)
    if cvc.low_quality:
        return np.zeros(len(u)), np.zeros(len(u)), 0, np.inf

    if m == 'deterministic':
        cvc.compute_deterministic()
    elif m == 'stochastic':
        cvc.compute_stochastic()
    velocity = cvc.get_velocity()
    r2 = cvc.get_r2()
    loss = cvc.get_loss()
    variance_velocity = (None if m == 'deterministic'
                         else cvc.get_variance_velocity())
    return velocity, variance_velocity, r2, loss


def velocity_chrom(adata_rna,
                   adata_atac=None,
                   gene_list=None,
                   mode='stochastic',
                   parallel=True,
                   n_jobs=None,
                   save_plot=False,
                   plot_dir=None,
                   rna_only=False,
                   extra_color_key=None,
                   min_r2=1e-2,
                   outlier=99.8,
                   n_pcs=30,
                   n_neighbors=30,
                   fig_size=(8, 6),
                   point_size=7
                   ):

    """Multi-omic steady-state model.

    This function incorporates chromatin accessibilities into RNA steady-state
    velocity.

    Parameters
    ----------
    adata_rna: :class:`~anndata.AnnData`
        RNA anndata object. Required fields: `Mu`, `Ms`, and `connectivities`.
    adata_atac: :class:`~anndata.AnnData` (default: `None`)
        ATAC anndata object. Required fields: `Mc`.
    gene_list: `str`,  list of `str` (default: highly variable genes)
        Genes to use for model fitting.
    mode: `str` (default: `'stochastic'`)
        Fitting method.
        `'stochastic'`: computing steady-state ratio with the first and second
        moments.
        `'deterministic'`: computing steady-state ratio with the first moments.
    parallel: `bool` (default: `True`)
        Whether to fit genes in a parallel fashion (recommended).
    n_jobs: `int` (default: available threads)
        Number of parallel jobs.
    save_plot: `bool` (default: `False`)
        Whether to save the fitted gene portrait figures as files. This will
        take some disk space.
    plot_dir: `str` (default: `plots` for multiome and `rna_plots` for
    RNA-only)
        Directory to save the plots.
    rna_only: `bool` (default: `False`)
        Whether to only use RNA for fitting (RNA velocity).
    extra_color_key: `str` (default: `None`)
        Extra color key used for plotting. Common choices are `leiden`,
        `celltype`, etc.
        The colors for each category must be present in one of anndatas, which
        can be pre-computed.
        with `scanpy.pl.scatter` function.
    min_r2: `float` (default: 1e-2)
        Minimum R-squared value for selecting velocity genes.
    outlier: `float` (default: 99.8)
        The percentile to mark as outlier that will be excluded when fitting
        the model.
    n_pcs: `int` (default: 30)
        Number of principal components to compute distance smoothing neighbors.
        This can be different from the one used for expression smoothing.
    n_neighbors: `int` (default: 30)
        Number of nearest neighbors for distance smoothing.
        This can be different from the one used for expression smoothing.
    fig_size: `tuple` (default: (8,6))
        Size of each figure when saved.
    point_size: `float` (default: 7)
        Marker point size for plotting.

    Returns
    -------
    fit_r2: `.var`
        R-squared of regression fit
    fit_loss: `.var`
        loss of model fit
    velo_s: `.layers`
        velocities in spliced space
    variance_velo_s: `.layers`
        variance velocities based on second moments in spliced space
    velo_s_genes: `.var`
        velocity genes
    velo_s_params: `.var`
        fitting arguments used
    ATAC: `.layers`
        KNN smoothed chromatin accessibilities copied from adata_atac
    """

    fit_args = {}
    fit_args['min_r2'] = min_r2
    fit_args['outlier'] = outlier
    fit_args['n_pcs'] = n_pcs
    fit_args['n_neighbors'] = n_neighbors
    fit_args['fig_size'] = list(fig_size)
    fit_args['point_size'] = point_size
    if mode == 'dynamical':
        logg.update('You do not need to run mv.velocity for chromatin '
                    'dynamical model', v=0)
        return
    elif mode == 'stochastic' or mode == 'deterministic':
        fit_args['mode'] = mode
    else:
        raise ValueError('Unknown mode. Must be either stochastic or '
                         'deterministic')

    all_genes = adata_rna.var_names
    if adata_atac is None:
        import anndata as ad
        rna_only = True
        adata_atac = ad.AnnData(X=np.ones(adata_rna.shape), obs=adata_rna.obs,
                                var=adata_rna.var)
        adata_atac.layers['Mc'] = np.ones(adata_rna.shape)
    if adata_rna.shape != adata_atac.shape:
        raise ValueError('Shape of RNA and ATAC adata objects do not match:'
                         f'{adata_rna.shape} {adata_atac.shape}')
    if not np.all(adata_rna.obs_names == adata_atac.obs_names):
        raise ValueError('obs_names of RNA and ATAC adata objects do not '
                         'match, please check if they are consistent')
    if not np.all(all_genes == adata_atac.var_names):
        raise ValueError('var_names of RNA and ATAC adata objects do not '
                         'match, please check if they are consistent')
    if extra_color_key is None:
        extra_color = None
    elif (isinstance(extra_color_key, str) and extra_color_key in adata_rna.obs
          and adata_rna.obs[extra_color_key].dtype.name == 'category'):
        ngroups = len(adata_rna.obs[extra_color_key].cat.categories)
        extra_color = adata_rna.obs[extra_color_key].cat.rename_categories(
            adata_rna.uns[extra_color_key+'_colors'][:ngroups]).to_numpy()
    elif (isinstance(extra_color_key, str)
          and extra_color_key in adata_atac.obs and
          adata_rna.obs[extra_color_key].dtype.name == 'category'):
        ngroups = len(adata_atac.obs[extra_color_key].cat.categories)
        extra_color = adata_atac.obs[extra_color_key].cat.rename_categories(
            adata_atac.uns[extra_color_key+'_colors'][:ngroups]).to_numpy()
    else:
        raise ValueError('Currently, extra_color_key must be a single string '
                         'of categories and available in adata obs, and its '
                         'colors can be found in adata uns')
    if ('connectivities' not in adata_rna.obsp.keys() or
            (adata_rna.obsp['connectivities'] > 0).sum(1).min()
            > (n_neighbors-1)):
        neighbors = Neighbors(adata_rna)
        neighbors.compute_neighbors(n_neighbors=n_neighbors,
                                    knn=True, n_pcs=n_pcs)
        rna_conn = neighbors.connectivities
    else:
        rna_conn = adata_rna.obsp['connectivities'].copy()
    rna_conn.setdiag(1)
    rna_conn = rna_conn.multiply(1.0 / rna_conn.sum(1)).tocsr()

    Mss, Mus = None, None
    if mode == 'stochastic':
        Mss, Mus = second_order_moments(adata_rna)

    if gene_list is None:
        if 'highly_variable' in adata_rna.var:
            gene_list = adata_rna.var_names[
                adata_rna.var['highly_variable']].values
        else:
            gene_list = adata_rna.var_names.values[
                (~np.isnan(np.asarray(adata_rna.layers['Mu'].sum(0))
                           .reshape(-1)
                           if sparse.issparse(adata_rna.layers['Mu'])
                           else np.sum(adata_rna.layers['Mu'], axis=0)))
                & (~np.isnan(np.asarray(adata_rna.layers['Ms'].sum(0))
                             .reshape(-1)
                             if sparse.issparse(adata_rna.layers['Ms'])
                             else np.sum(adata_rna.layers['Ms'], axis=0)))
                & (~np.isnan(np.asarray(adata_atac.layers['Mc'].sum(0))
                             .reshape(-1)
                             if sparse.issparse(adata_atac.layers['Mc'])
                             else np.sum(adata_atac.layers['Mc'], axis=0)))]
    elif isinstance(gene_list, (list, np.ndarray, pd.Index, pd.Series)):
        gene_list = np.array([x for x in gene_list if x in all_genes])
    elif isinstance(gene_list, str):
        gene_list = np.array([gene_list]) if gene_list in all_genes else []
    else:
        raise ValueError('Invalid gene list, must be one of (str, np.ndarray,'
                         ' pd.Index, pd.Series)')
    gn = len(gene_list)
    if gn == 0:
        raise ValueError('None of the genes specified are in the adata object')
    
    logg.update(f'{gn} genes will be fitted', v=1)

    velo_s = np.zeros((adata_rna.n_obs, gn))
    variance_velo_s = np.zeros((adata_rna.n_obs, gn))
    r2s = np.zeros(gn)
    losses = np.zeros(gn)

    u_mat = (adata_rna[:, gene_list].layers['Mu'].A
             if sparse.issparse(adata_rna.layers['Mu'])
             else adata_rna[:, gene_list].layers['Mu'])
    s_mat = (adata_rna[:, gene_list].layers['Ms'].A
             if sparse.issparse(adata_rna.layers['Ms'])
             else adata_rna[:, gene_list].layers['Ms'])
    c_mat = (adata_atac[:, gene_list].layers['Mc'].A
             if sparse.issparse(adata_atac.layers['Mc'])
             else adata_atac[:, gene_list].layers['Mc'])
    if parallel:
        if (n_jobs is None or not isinstance(n_jobs, int) or
                n_jobs < 0 or n_jobs > os.cpu_count()):
            n_jobs = os.cpu_count()
        if n_jobs > gn:
            n_jobs = gn
        batches = -(-gn // n_jobs)
        if n_jobs > 1:
            logg.update(f'running {n_jobs} jobs in parallel', v=1)
    else:
        n_jobs = 1
        batches = gn
    if n_jobs == 1:
        parallel = False

    pbar = tqdm(total=gn)
    for group in range(batches):
        gene_indices = range(group * n_jobs, np.min([gn, (group+1) * n_jobs]))
        if parallel:
            verb = 51 if settings.VERBOSITY >= 2 else 0

            res = Parallel(n_jobs=n_jobs, backend='loky', verbose=verb)(
                delayed(regress_func)(
                    c_mat[:, i],
                    u_mat[:, i],
                    s_mat[:, i],
                    None if mode == 'deterministic' else Mss[:, i],
                    None if mode == 'deterministic' else Mus[:, i],
                    mode,
                    save_plot,
                    plot_dir,
                    fit_args,
                    gene_list[i],
                    rna_only,
                    extra_color)
                for i in gene_indices)

            for i, r in zip(gene_indices, res):
                velocity, variance_velocity, r2, loss = r
                r2s[i] = r2
                losses[i] = loss
                velo_s[:, i] = smooth_scale(rna_conn, velocity)
                if mode == 'stochastic':
                    variance_velo_s[:, i] = smooth_scale(rna_conn,
                                                         variance_velocity)

        else:
            i = group
            gene = gene_list[i]
            logg.update(f'@@@@@fitting {gene}', v=1)
            velocity, variance_velocity, r2, loss = \
                regress_func(c_mat[:, i],
                             u_mat[:, i],
                             s_mat[:, i],
                             None
                             if mode == 'deterministic' else Mss[:, i],
                             None if mode == 'deterministic' else Mus[:, i],
                             mode,
                             save_plot,
                             plot_dir,
                             fit_args,
                             gene_list[i],
                             rna_only,
                             extra_color)
            r2s[i] = r2
            losses[i] = loss
            velo_s[:, i] = smooth_scale(rna_conn, velocity)
            if mode == 'stochastic':
                variance_velo_s[:, i] = smooth_scale(rna_conn,
                                                     variance_velocity)
        pbar.update(len(gene_indices))
    pbar.close()

    filt = losses != np.inf
    if np.sum(filt) == 0:
        raise ValueError('None of the genes were fitted due to low quality, '
                         'not returning')
    adata_copy = adata_rna[:, gene_list[filt]].copy()
    adata_copy.layers['ATAC'] = c_mat[:, filt]
    adata_copy.var['fit_loss'] = losses[filt]
    adata_copy.var['fit_r2'] = r2s[filt]
    adata_copy.layers['velo_s'] = velo_s[:, filt]
    if mode == 'stochastic':
        adata_copy.layers['variance_velo_s'] = variance_velo_s[:, filt]
    v_genes = adata_copy.var['fit_r2'] >= min_r2
    adata_copy.var['velo_s_genes'] = v_genes
    adata_copy.uns['velo_s_params'] = {'mode': mode, 'fit_offset': False,
                                       'perc': 95}
    adata_copy.uns['velo_s_params'].update(fit_args)
    adata_copy.obsp['_RNA_conn'] = rna_conn
    return adata_copy


def smooth_scale(conn, vector):
    max_to = np.max(vector)
    min_to = np.min(vector)
    v = conn.dot(vector.T).T
    max_from = np.max(v)
    min_from = np.min(v)
    res = ((v - min_from) * (max_to - min_to) / (max_from - min_from)) + min_to
    return res


###############################################################################
# The following functions are taken directly from scVelo preprocessing
# [Bergen et al., 2020] (https://github.com/theislab/scvelo)
###############################################################################

def select_connectivities(connectivities, n_neighbors=None):
    C = connectivities.copy()
    n_counts = (C > 0).sum(1).A1 if issparse(C) else (C > 0).sum(1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(),
                                                       n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = C.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[::-1][n_neighbors:]
        dat[rm_idx] = 0
    C.eliminate_zeros()
    return C


def get_neighs(adata, mode="distances"):
    if hasattr(adata, "obsp") and mode in adata.obsp.keys():
        return adata.obsp[mode]
    elif "neighbors" in adata.uns.keys() and mode in adata.uns["neighbors"]:
        return adata.uns["neighbors"][mode]
    else:
        raise ValueError("The selected mode is not valid.")


def get_n_neighs(adata):
    return (adata.uns.get("neighbors", {}).get("params", {})
            .get("n_neighbors", 0))


def get_connectivities(adata, mode="connectivities", n_neighbors=None,
                       recurse_neighbors=False):
    if "neighbors" in adata.uns.keys():
        C = get_neighs(adata, mode)
        if n_neighbors is not None and n_neighbors < get_n_neighs(adata):
            if mode == "connectivities":
                C = select_connectivities(C, n_neighbors)
            else:
                C = select_distances(C, n_neighbors)
        connectivities = C > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            connectivities.setdiag(1)
            if recurse_neighbors:
                connectivities += connectivities.dot(connectivities * 0.5)
                connectivities.data = np.clip(connectivities.data, 0, 1)
            connectivities = connectivities.multiply(1.0 /
                                                     connectivities.sum(1))
        return connectivities.tocsr().astype(np.float32)
    else:
        return None


def second_order_moments(adata, adjusted=False):
    """Computes second order moments for stochastic velocity estimation.
    Arguments
    ---------
    adata: `AnnData`
        Annotated data matrix.
    Returns
    -------
    Mss: Second order moments for spliced abundances
    Mus: Second order moments for spliced with unspliced abundances
    """

    if "neighbors" not in adata.uns:
        raise ValueError(
            "You need to run `pp.neighbors` first to compute a neighborhood "
            "graph."
        )

    connectivities = get_connectivities(adata)
    s, u = csr_matrix(adata.layers["spliced"]), \
        csr_matrix(adata.layers["unspliced"])
    if s.shape[0] == 1:
        s, u = s.T, u.T
    Mss = csr_matrix.dot(connectivities, s.multiply(s)).astype(np.float32).A
    Mus = csr_matrix.dot(connectivities, s.multiply(u)).astype(np.float32).A
    if adjusted:
        Mss = 2 * Mss - adata.layers["Ms"].reshape(Mss.shape)
        Mus = 2 * Mus - adata.layers["Mu"].reshape(Mus.shape)
    return Mss, Mus
