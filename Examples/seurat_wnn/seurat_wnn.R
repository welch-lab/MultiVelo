# MultiVelo Seurat WNN Demo
# The procedure mostly follows Seurat tutorial: https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis.html
# Note that these preprocessing steps are what we found to be the best for the 10X mouse brain demo. These steps may need to be modified for some datasets to give the best performance.
# For example, we have found that regressing out cell cycle is helpful in some cases.
# We used libblas 3.9.1 and liblapack 3.9.1 to generate the results in the paper. The visualizations differ slightly if you use a different version.
# To reproduce the exact results from the paper, please match the libblas and liblapack versions or use the supplied WNN files on GitHub.

library(Seurat)
library(Signac)

# read in expression and accessbility data
brain.data <- Read10X(data.dir = "../outs/filtered_feature_bc_matrix/")

# subset for the same cells in the jointly filtered anndata object
barcodes <- read.delim("../filtered_cells.txt", header = F, stringsAsFactors = F)$V1

# preprocess RNA
brain <- CreateSeuratObject(counts = brain.data$`Gene Expression`[,barcodes])
brain <- NormalizeData(brain)
brain <- FindVariableFeatures(brain)
brain <- ScaleData(brain, do.scale = F) # not scaled for consistency with scVelo (optionally, use SCTransform)
brain <- RunPCA(brain, verbose = FALSE)
brain <- RunUMAP(brain, dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_') # optional

# preprocess ATAC
brain[["ATAC"]] <- CreateAssayObject(counts = brain.data$`Peaks`[,barcodes], min.cells = 1)
DefaultAssay(brain) <- "ATAC"
brain <- RunTFIDF(brain)
brain <- FindTopFeatures(brain, min.cutoff = 'q0')
brain <- RunSVD(brain)
brain <- RunUMAP(brain, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_") # optional

# find weighted nearest neighbors
brain <- FindMultiModalNeighbors(brain, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50), k.nn = 50)
brain <- RunUMAP(brain, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_") # optional

# extract neighborhood graph
nn_idx <- brain@neighbors$weighted.nn@nn.idx
nn_dist <- brain@neighbors$weighted.nn@nn.dist
nn_cells <- brain@neighbors$weighted.nn@cell.names

# save neighborhood graph
write.table(nn_idx, "nn_idx.txt", sep = ',', row.names = F, col.names = F, quote = F)
write.table(nn_dist, "nn_dist.txt", sep = ',', row.names = F, col.names = F, quote = F)
write.table(nn_cells, "nn_cells.txt", sep = ',', row.names = F, col.names = F, quote = F)

# save sessionInfo for reproducibility
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
