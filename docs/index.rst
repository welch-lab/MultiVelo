|Stars| |PyPI| |PyPIDownloads| |Bioconda| |Docs|

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


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   usage
   api

.. toctree::
   :caption: Demo
   :maxdepth: 1
   :hidden:

   MultiVelo_Demo

.. toctree::
   :caption: Template
   :maxdepth: 1
   :hidden:

   MultiVelo_Template


.. |Stars| image:: https://img.shields.io/github/stars/welch-lab/multivelo?logo=GitHub&color=yellow
   :target: https://github.com/welch-lab/multivelo/stargazers

.. |PyPI| image:: https://img.shields.io/pypi/v/multivelo?logo=PyPI
   :target: https://pypi.org/project/multivelo

.. |PyPIDownloads| image:: https://pepy.tech/badge/multivelo
   :target: https://pepy.tech/project/multivelo

.. |Bioconda| image:: https://img.shields.io/conda/dn/bioconda/multivelo?logo=Anaconda
   :target: https://anaconda.org/bioconda/multivelo

.. |Docs| image:: https://readthedocs.org/projects/multivelo/badge/?version=latest
   :target: https://multivelo.readthedocs.io
