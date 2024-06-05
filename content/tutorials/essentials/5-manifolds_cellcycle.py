# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Manifolds: the cell cycle

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac

import numpy as np

# %% [markdown]
# We'll use a control dataset, containing cells in which mCherry was overexpressed

# %%
adata = la.data.load_control()

# %% tags=["hide-input", "remove-output"]
import scanpy as sc

adata.raw = adata

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

sc.pp.combat(adata)
sc.pp.pca(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["gene_overexpressed", "batch", "phase"])
sc.pl.pca(adata, color=["gene_overexpressed", "batch", "phase"])

# %% [markdown]
# ## Cell cycle

# %%
import laflow as laf

# %%
flow = laf.TemporaryFlow()
dataset = lac.transcriptome.create_dataset_from_adata(adata, "dataset", flow)
model = lac.Model("model", flow, dataset = dataset)

# %%
model.add_process(lac.transcriptome.Transcriptome())
model.add_process(lac.cell.batch.Given())
model.add_process(lac.cell.cellcycle.Latent(), pretrain = True)

# %%
model.transcriptome.add_effect(lac.transcriptome.effects.batch.Linear(batch = model.batch))
model.transcriptome.add_effect(lac.transcriptome.effects.cellcycle.Spline(cellcycle = model.cellcycle, subset_genes = True))
model.transcriptome.add_effect(lac.transcriptome.effects.cellcycle.Weights(cellcycle = model.cellcycle))

# %%
model.infer()

# %%
model.cellcycle.interpret(force = True)

# %%
adata = model.dataset.adata

# %%
import matplotlib as mpl
import seaborn as sns

# %%
adata.obs["cellcycle"] = la.domains.circular.mean(
    model.cellcycle.observed.samples[model.cellcycle.observed.output], "sample"
).to_pandas()
sc.pl.pca(adata, color=["cellcycle", "phase", "batch"])
sc.pl.umap(adata, color=["cellcycle", "phase", "batch"])
fig, (ax0, ax1) = mpl.pyplot.subplots(1, 2, figsize=(10, 5))
ax = sns.stripplot(x=adata.obs["cellcycle"], y=adata.obs["phase"], size=1, ax=ax0)
ax = sns.stripplot(x=adata.obs["cellcycle"], y=adata.obs["batch"], size=1, ax=ax1)

# %%
model.transcriptome.cellcycle.interpret()

# %%
gene_ids = list(set(dataset.transcriptome.var.index) & set(lac.transcriptome.effects.cellcycle.cellcycle_genes.get_cellcycle_genes("mm")["gene"]))

# %%
model.transcriptome.cellcycle.conditional.plot_features_heatmap(gene_ids, transform = None)

# %% [markdown]
# ## Transfering manifolds

# %%

# %%
