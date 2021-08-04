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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Discrete distributions

# %%
from IPython import get_ipython
if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import torch
import math

import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import collections

import latenta as la
la.logger.setLevel("INFO")

# %%
cells = la.Dim(pd.Series(range(3), name = "cell").astype(str))
genes = la.Dim(pd.Series(range(5), name = "gene").astype(str))
celltypes = la.Dim(pd.Series(range(4), name = "celltype").astype(str))

# %%
probs_value = pd.DataFrame(0.1, index = cells.index, columns = celltypes.index)
probs_value = probs_value / probs_value.values.sum(0, keepdims = True)

# %%
dist = la.distributions.OneHotCategorical(0.1, definition = la.Definition([cells, celltypes]))

# %%
dist.reset_recursive()
dist.run_recursive()
dist.value

# %% [markdown]
# Different component dim (not the end)

# %%
probs_value = pd.DataFrame(0.1, index = cells.index, columns = celltypes.index)
probs_value = probs_value / probs_value.values.sum(0, keepdims = True)

# %%
dist = la.distributions.OneHotCategorical(0.1, definition = la.Definition([cells, celltypes]), component_dim=cells)

# %%
dist.reset_recursive()
dist.run_recursive()
dist.value

# %% [markdown]
# More than 2 dimensions

# %%
probs_value = np.ones((cells.size, celltypes.size, genes.size))
probs_value = probs_value / probs_value.sum(0, keepdims = True)

# %%
dist = la.distributions.OneHotCategorical(0.1, definition = la.Definition([cells, genes, celltypes]), component_dim=cells)

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.likelihood)

# %%
