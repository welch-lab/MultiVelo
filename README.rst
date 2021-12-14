MultiVelo - Multi-omic extension of single-cell RNA velocity
============================================================

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

Install through PyPI: `pip install multivelo`


An example to run MultiVelo can be found in `multivelo_demo <https://github.com/welch-lab/MultiVelo/tree/main/multivelo_demo>`_

We will use the embryonic E18 mouse brain from 10X Multiome as an example (`jupyter notebook <https://github.com/welch-lab/MultiVelo/tree/main/multivelo_demo/MultiVelo_Demo.ipynb>`_).

If you would like to run the example yourself. CellRanger output files can be downloaded from 
`10X website <https://www.10xgenomics.com/resources/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0>`_. 
Crucially, the filtered feature barcode matrix folder, ATAC peak annotations TSV, and the feature 
linkage BEDPE file in the secondary analysis outputs folder will be needed in this demo.

Quantified unspliced and spliced counts from Velocyto can be downloaded from MultiVelo Github page.

We provide the cell annotations for this dataset in "cell_annotations.tsv" on the Github page. 
(To download from Github, click on the file, then click "Raw" on the top right corner. 
If it opens in your browser, you can download the page as a text file.)

Weighted nearest neighbors from Seurat can be downloaded from Github folder "seurat_wnn", 
which contains three files: "nn_cells.txt", "nn_dist.txt", and "nn_idx.txt". The R script used 
to generate such files can also be found on the Github page (to be added).
