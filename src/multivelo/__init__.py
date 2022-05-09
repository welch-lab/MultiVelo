from .auxiliary import (
    aggregate_peaks_10x,
    tfidf_norm,
    knn_smooth_chrom,
    calculate_qc_metrics,
    ellipse_fit,
    compute_quantile_scores,
    cluster_by_quantile
)

from .dynamical_chrom_func import (
    ChromatinDynamical,
    recover_dynamics_chrom,
    set_velocity_genes,
    velocity_graph,
    velocity_embedding_stream,
    latent_time,
    LRT_decoupling,
    likelihood_plot,
    pie_summary,
    switch_time_summary,
    dynamic_plot,
    scatter_plot
)

from .steady_chrom_func import (
    ChromatinVelocity,
    velocity_chrom
)

from importlib.metadata import version
__version__ = version(__name__)
