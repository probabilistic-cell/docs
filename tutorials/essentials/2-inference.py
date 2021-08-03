# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inference

# %%
import latenta as la
import scanpy as sc

# %% [markdown]
# We'll use the same dataset as [before](./1-variables)...

# %% 
adata = sc.datasets.pbmc3k()

sc.pp.filter_cells(adata, min_counts = 100)

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.filter_genes_dispersion(adata)

sc.pp.pca(adata)
sc.pp.neighbors(adata)

# %% [markdown]
# and also the same model:

