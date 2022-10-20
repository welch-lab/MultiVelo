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

Install through PyPI: **pip install multivelo**

**New**: Now available on Bioconda. Install with **conda install -c bioconda multivelo** or **mamba install -c bioconda multivelo**

**New**: `ReadTheDocs <https://multivelo.readthedocs.io/en/latest/>`_ page is up. You can find detailed parameter descriptions and tutorials on the website.

A tutorial showing how to run MultiVelo can be found in `multivelo_demo <https://github.com/welch-lab/MultiVelo/tree/main/multivelo_demo>`_.

We use the embryonic E18 mouse brain from 10X Multiome as an example (`jupyter notebook <https://github.com/welch-lab/MultiVelo/tree/main/multivelo_demo/MultiVelo_Demo.ipynb>`_).

CellRanger output files can be downloaded from 
`10X website <https://www.10xgenomics.com/resources/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0>`_. 
Crucially, the filtered feature barcode matrix folder, ATAC peak annotations TSV, and the feature 
linkage BEDPE file in the secondary analysis outputs folder will be needed in this demo.

You can download the processed data that we used for this analysis if you want to run the example yourself. 
Unspliced and spliced counts, as well as cell type annotations can be downloaded from the MultiVelo GitHub page. 
We provide the cell annotations for this dataset in "cell_annotations.tsv". 
We also provide the nearest neighbor graph used to smooth chromatin accessibility values in the GitHub folder "seurat_wnn", 
which contains a zip file of three files: "nn_cells.txt", "nn_dist.txt", and "nn_idx.txt". Please unzip the archive after downloading. 
The R script used to generate these files can also be found in the same folder.

| Citation:
| Li, C., Virgilio, M.C., Collins, K.L. & Welch J.D. Multi-omic single-cell velocity models epigenome–transcriptome interactions and improves cell fate prediction. *Nat Biotechnol* (2022).
