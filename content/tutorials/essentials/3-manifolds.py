# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Manifolds

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables)...

# %%
adata = la.data.load_myod1()

# %% tags=["hide-input", "remove-output"]
import scanpy as sc

adata.raw = adata

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

sc.pp.combat(adata)
sc.pp.pca(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])

# %%
sc.pl.umap(adata, color=["gene_overexpressed", "batch", "log_overexpression"])

# %% [markdown]
# So far, we have focused on models that basically look like this:

# %% [markdown]
# ![](_static/previous.svg)

# %% [markdown]
# But what if our cellular variables are also unknown, i.e. latent?

# %% [markdown]
# ![](_static/now.svg)

# %% [markdown]
# In this case, apart from learning a latent space about genes, we _at the same time_ also try to learn a latent space on cells, also known as the "cellular manifold". There are many different ways to represent this manifold, and how to do this depends on what we know about the cells and what we try to accomplish:
#
# - **embedding**: Each cell is placed in an n-dimensional space, with only a few restrictions, such as the cell positions are all centered around 0. This space is very flexible, but not interpretable. Essentially, it is only interpretable when converted to a 2D representation and viewed by the human eye. Related to this, this space is typically far removed from biological reality. A cell's state does not "float" in an n-dimensional space.
# - **scalar** (differentiation, cell cycle, ...): Each cell is placed in a one-dimensional space, with a single pseudotime.

# %% [markdown]
# While many manifold models are relatively easy to implement, the main difficulty lies in the interpretability. Especially when different **cellular processes** are happening at the same time in a cell, a latent variable will try to explain all of them. What is therefore often required is the inclusion of prior knowledge that can help with disentangling different cellular processes.

# %% [markdown]
# ## Differentiation: Inferring a dominant scalar latent variable

# %% [markdown]
# Just by looking at the 2d representation, it already becomes obvious that there are two dominant processes going on in the cell: differentiation (in this case to myocytes) and the cell cycle. Moreover, we can simply remove (or reduce) the effect of the cell cycle by removing the cycling cells.

# %%
cellcycle_genes = lac.cell.cellcycle.get_cellcycle_genes()

# %%
sc.tl.score_genes_cell_cycle(adata, s_genes = cellcycle_genes.query("phase == 'S'")["gene"].tolist(), g2m_genes = cellcycle_genes.query("phase == 'G2/M'")["gene"].tolist())

# %%
sc.pl.umap(adata, color=["gene_overexpressed", "batch", "log_overexpression", "phase"])

# %%
adata_oi = adata[adata.obs["phase"] == "G1"].copy()

# %%
transcriptome = lac.transcriptome.Transcriptome.from_adata(adata_oi)

# %%
transcriptome.plot()

# %%
cells = transcriptome["cell"]

# %%
differentiation = la.Latent(la.distributions.Uniform(), definition = la.Definition([cells]), label = "differentiation")

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.differentiation = la.links.scalar.Spline(differentiation, output = foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
differentiation_observed = la.posterior.Observed(differentiation)
differentiation_observed.sample(10)

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color = ["differentiation"])

# %% [markdown]
# Even though the differentiation is very dominant, the algorithm still used about half of the latent space to explain some heterogeneity in the control cells. This can have various reasons, for example .

# %% [markdown]
# In this case, we can easily fix this by including some external information. Namely, we know which cells were not perturbed, and force those cells to differentiation values close to 0 by specifying an appropriate prior distribution.

# %%
import pandas as pd

# %%
differentiation_p = la.distributions.Beta(
    concentration0 = la.Fixed(pd.Series([1., 100.], index = ["Myod1", 'mCherry'])[adata_oi.obs["gene_overexpressed"]].values, definition = la.Definition([cells])),
    concentration1 = la.Fixed(pd.Series([1., 1.], index = ["Myod1", 'mCherry'])[adata_oi.obs["gene_overexpressed"]].values, definition = la.Definition([cells]))
)

# %%
differentiation = la.Latent(differentiation_p, definition = la.Definition([cells]), label = "differentiation")

# %%
# encoder = la.amortization.Encoder(la.Fixed(transcriptome.loader, definition = transcriptome), differentiation, pretrain = True)

# %%
differentiation.plot()

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.differentiation = la.links.scalar.Spline(differentiation, output = foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
differentiation_observed = la.posterior.Observed(differentiation)
differentiation_observed.sample(10)

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color = ["differentiation", "batch"])

# %%
sc.pl.umap(adata_oi, color = adata.var.reset_index().set_index("symbol").loc[["Myog"]].gene)

# %%
differentiation_causal = la.posterior.scalar.ScalarVectorCausal(
    differentiation, transcriptome, observed = differentiation_observed, interpretable = transcriptome.p.mu.expression
)
differentiation_causal.sample(10)

# %% [markdown]
# This posterior also contains samples, but now for some pseudocells with each a distinct value for the `overexpression` variable.

# %% [markdown]
# Depending on the type of causal posterior, you can plot the outcome. The {class}`~latenta.posterior.scalar.ScalarVectorCausal` can for example plot each individual _feature_ across all cells (in this case gene):

# %%
differentiation_causal.sample_empirical()
differentiation_causal.sample_random()

# %%
differentiation_causal.plot_features();

# %%
