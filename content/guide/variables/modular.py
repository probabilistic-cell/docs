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
# # Modular variables

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import torch

import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import collections

import latenta as la

la.logger.setLevel("INFO")

# %% [markdown]
# ## Simple example

# %%
n_cells = 50
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(cell_ids, name="cell"))

x = la.Fixed(pd.Series(np.random.uniform(0, 3, n_cells), index=cells.index), label="x")


# %%
n_genes = 100
genes = la.Dim(pd.Series([str(i) for i in range(n_genes)]), name="gene")


def random_coefficient(n_genes):
    return (
        np.random.choice([-1, 1], n_genes)
        * np.random.normal(3.0, 1.0, n_genes)
        * (np.random.random(n_genes) > 0.5)
    )


slope = la.Fixed(
    pd.Series(random_coefficient(n_genes), index=genes.index), label="slope"
)
intercept = la.Fixed(
    pd.Series(random_coefficient(n_genes), index=genes.index), label="intercept"
)
final = la.Fixed(
    pd.Series(random_coefficient(n_genes), index=genes.index), label="final"
)

slope_subset = la.Fixed(
    pd.Series(random_coefficient(10), index=genes.index[:10]), label="slope_subset"
)


# %%
y_linear = la.links.scalar.Linear(x=x, a=slope)
y = la.modular.Additive(
    linear=y_linear, definition=y_linear.value_definition, subsettable={genes}
)

# %% [markdown]
# We can add extra terms:

# %%
y_sigmoid = la.links.scalar.Sigmoid(x=x, a=final)
y.sigmoid = y_sigmoid

y_exponential = la.links.scalar.Exp(x=x, a=final)
y.exponential = y_exponential

# %% [markdown]
# The terms can have a subset of the modular variable's dimensions:

# %%
y.simply_x = x
y.simply_slope = slope


# %% [markdown]
# In fact, a terms dimension can be a subset, as long as this dimension was registered as "subsettable" when defining the modular variable:

# %%
y_linear_subset = la.links.scalar.Linear(x=x, a=slope_subset)
y.linear_subset = y_linear_subset
y_linear.prior_xr().shape, y_linear_subset.prior_xr().shape

# %% [markdown]
# We can also remove terms

# %%
del y.exponential

# %%
scale = la.Fixed(
    pd.Series(np.random.uniform(1.0, 1.2, n_genes), index=genes.index), label="scale"
)
dist = la.distributions.Normal(loc=y, scale=scale)

# %%
model_gs = la.Root(dist = dist, label="ground truth", symbol="gs")
model_gs.plot()

# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)

# %% [markdown]
# ## Dynamic definition

# %% [markdown]
# Currently, modular variables are one of the few variables with a "dynamic" definition, in the sense that they can get new dimensions for free.

# %%
n_cells = 50
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(range(n_cells), name="cell").astype(str))

n_genes = 100
genes = la.Dim(pd.Series(range(n_genes), name="gene").astype(str))

# %%
x = la.Fixed(1., definition = [cells])
y = la.Fixed(1., definition = [genes])

# %%
z = la.modular.Additive(definition = [cells])
z.x = x

# %%
z.y = y

# %%
z

# %%
z.prior()

# %% [markdown]
# ## Types of modular variables

# %% [markdown]
# ### Multiplicative

# %%
y = la.modular.Multiplicative(
    linear=y_linear,
    linear_subset=y_linear_subset,
    definition=y_linear.value_definition,
    subsettable={genes},
    label = "multi"
)

# %%
y.plot()

# %%
y.prior_pd()

# %%

# %%
