.. automodule:: multivelo

API
===

Import MultiVelo as::

    import multivelo as mv


Preprocessing
------------------

.. autosummary::
   :toctree: .

   aggregate_peaks_10x
   tfidf_norm
   knn_smooth_chrom
   calculate_qc_metrics

Tools
------------------

.. autosummary::
   :toctree: .

   recover_dynamics_chrom
   velocity_chrom
   set_velocity_genes
   velocity_graph
   velocity_embedding_stream
   latent_time
   LRT_decoupling

Plotting
------------------

.. autosummary::
   :toctree: .

   likelihood_plot
   pie_summary
   switch_time_summary
   dynamic_plot
   scatter_plot
