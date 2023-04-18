# An example to run DTW time delay analysis across modalities.

library(tidyverse)
library(dtw)
library(zoo)

gene = 'EGR1'
bins = 20

# input file contains three columns: latent time, motif accessibility, and TF gene expression
res = read.delim(paste0(gene, '_res.txt'), sep=' ', header = F, stringsAsFactors = F)
colnames(res) = c('t', 'm', 'r')
res = res[order(res$t),]

res$t = res$t * bins
res$window = floor(res$t / 1)
res$window[res$window==bins] = bins-1
res2 = aggregate(res, by=list(res$window), FUN=mean)

rv = res2$r - min(res2$r)
rv = c(rv[1], rv, rv[length(rv)])
rv = rollmean(rv, 3)
rv[1] = 0
rv[length(rv)] = 0
rv = rv / max(rv)
mv = res2$m - min(res2$m)
mv = c(mv[1], mv, mv[length(mv)])
mv = rollmean(mv, 3)
mv[1] = 0
mv[length(mv)] = 0
mv = mv / max(mv)

# compute dtw
dtw.res = dtw(rv, mv, keep.internals = T)
dtwPlotTwoWay(dtw.res, col=c('magenta', '#f7ba41'), lwd=3, xlab='latent time', ylab='accessibility and expression')
