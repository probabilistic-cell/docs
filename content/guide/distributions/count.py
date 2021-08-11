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

# %% [markdown]
# # Count distributions

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
genes = la.Dim(pd.Series(range(4), name = "gene").astype(str))

# %% [markdown]
# ## Negative Binomial

# %%
dist = la.distributions.NegativeBinomial1(
    definition = la.Definition([cells])
)

# %%
dist.reset()
dist.run()
print(dist.value)
print(dist.likelihood)
print(dist.log_prob(torch.tensor([0., 1., 2.])))

# %%
dist = la.distributions.NegativeBinomial1(
    logits = la.Fixed(math.log(100) - math.log(1), definition = la.Definition([cells])),
    total_count = la.Fixed(1.)
)

# %%
dist.plot()

# %%
dist.reset()
dist.run()
print(dist.value)
print(dist.likelihood)
print(dist.log_prob(torch.tensor([10., 100., 200.])))

# %%
dist = la.distributions.NegativeBinomial2(
    mu = la.Fixed(100., definition = la.Definition([cells])),
    dispersion = la.Fixed(1.)
)

# %%
dist.plot()

# %%
dist.reset()
dist.run()
print(dist.value)
print(dist.likelihood)
print(dist.log_prob(torch.tensor([10., 100., 200.])))
