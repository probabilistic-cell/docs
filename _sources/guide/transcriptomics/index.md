# Transcriptomics

## Q&A

(why-not-just-provide-normalized-data)=
### Why don't we just provide normalized data? 

You may be inclined to provide normalized data to latenta, in which you already removed co-variates and batch effects. While possible in principle, it has several disadvantages:

- **Flexibility**: Normalization always comes with some assumptions about which factors are technical versus biological, which can be easily violated. The number of mRNAs in a cell is for example well known to be associated with cell size and cell cycle stage, both of which may be biologically relevant. A typical library size normalization would remove this information.

- **Statistics**: Normalization typically removes (or at least distorts) any statistical information in the data. Although this can be resolved in some exceptional cases (e.g. limma/voom), there is no solution that is generalizable to any moderately complex models.

- **Scalability**: Not normalizing allows us to retain our data in sparse format. We might not be so lucky for other modalities, but it's a major advantage when working on memory-constrained systems such as a GPU.