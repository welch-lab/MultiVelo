Usage
=====

.. _installation:

Installation
------------

To use MultiVelo, first install it using pip or conda:

.. code-block:: console

   $ pip install multivelo

or

.. code-block:: console

   $ conda install -c bioconda multivelo

Demo
----

A tutorial showing how to run MultiVelo can be found in `multivelo_demo <https://github.com/welch-lab/MultiVelo/tree/main/multivelo_demo>`_.

We use the embryonic E18 mouse brain from 10X Multiome as an example.

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

