|PyPI| |PyPIDownloads| |Docs|

MultiVelo - Velocity Inference from Single-Cell Multi-Omic Data
===============================================================

Single-cell multi-omic datasets, in which multiple molecular modalities are profiled 
within the same cell, provide a unique opportunity to discover the interplay between 
cellular epigenomic and transcriptomic changes. To realize this potential, we developed 
**MultiVelo**, a mechanistic model of gene expression that extends the popular RNA velocity 
framework by incorporating epigenomic data.

MultiVelo uses a probabilistic latent variable model to estimate the switch time and rate 
parameters of gene regulation, providing a quantitative summary of the temporal relationship 
between epigenomic and transcriptomic changes. Fitting MultiVelo on single-cell multi-omic 
datasets revealed two distinct mechanisms of regulation by chromatin accessibility, quantified 
the degree of concordance or discordance between transcriptomic and epigenomic states within 
each cell, and inferred the lengths of time lags between transcriptomic and epigenomic changes.

Check out the :doc:`usage` section for further information.


Contents
--------

.. toctree::

   usage
   api
