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
# # Distributions

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
# ## Basic distributions

# %%
dist = la.distributions.Normal(loc = 1.)

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value, dist.likelihood)

# %%
dist.prior_mean

# %% [markdown]
# ### Automatic broadcasting

# %%
dist = la.distributions.Normal(definition = la.Definition([cells]))

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.likelihood)

# %%
dist = la.distributions.Normal(loc = la.Fixed(10., definition = la.Definition([cells])))

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.likelihood)

# %%
dist = la.distributions.Normal(loc = la.Fixed(10., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])))

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.likelihood)

# %%
dist.redefine(la.Definition([cells, genes]))
dist.redefine(la.Definition([genes, cells]))

# %% [markdown]
# ### Log probability with new dimensions

# %%
dist_1 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])))
dist_2 = la.distributions.Normal(loc = la.Fixed(np.linspace(-1, 1., len(cells)), definition = la.Definition([cells])), definition = dist_1.clean)
dist_3 = la.distributions.Normal(loc = la.Fixed(np.linspace(-1, 1., len(genes)), definition = la.Definition([genes])), definition = dist_1.clean)

# %%
dist_1.reset_recursive()
dist_1.run_recursive()

dist_2.reset_recursive()
dist_2.run_recursive()

dist_3.reset_recursive()
dist_3.run_recursive()

# %%
dist_2.likelihood

# %%
(dist_2.likelihood == dist_2.log_prob(dist_2.value)).all()

# %%
dist_2.log_prob(dist_1.value)

# %%
dist_3.likelihood

# %%
(dist_3.likelihood == dist_3.log_prob(dist_3.value)).all()

# %%
dist_3.log_prob(dist_1.value)

# %% [markdown]
# ### Dependent dimensions

# %% [markdown]
# We can mark some dimensions as dependent dimensions:

# %%
dist_1 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])), dependent_dims = {cells})
dist_2 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])))

# %%
dist_1.reset_recursive()
dist_1.run_recursive()

dist_2.reset_recursive()
dist_2.run_recursive()

# %%
dist_1.likelihood

# %%
dist_2.likelihood

# %% [markdown]
# In contrast to pytorch and tensorflow, we do not assume that event dimensions are at the end of the tensor. They can be anywhere

# %%
dist_3 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])), dependent_dims = {cells})
dist_4 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])), dependent_dims = {genes})

# %%
dist_3.reset_recursive()
dist_3.run_recursive()

dist_4.reset_recursive()
dist_4.run_recursive()

# %%
dist_3.likelihood

# %%
dist_4.likelihood

# %% [markdown]
# You can always know which dimensions are dependent using `dist.event_dims`

# %%
dist.event_dims

# %% [markdown]
# ## Event-based distributions

# %% [markdown]
# Some distributions have default dependent dimensions

# %%
dist = la.distributions.Dirichlet(concentration = la.Fixed(1., definition = la.Definition([cells])))

# %%
dist.event_dims

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.likelihood)

# %%
value = torch.ones(dist.value_definition.shape)
value = value / value.sum(0, keepdim = True)
assert torch.isclose(torch.exp(dist.log_prob(value)), torch.tensor(2.))

# %% [markdown]
# We can of course add multiple dimensions to such a distribution. By default (as per tensorflow/torch standards), the dependent dimension will be the last one:

# %%
dist = la.distributions.Dirichlet(concentration = la.Fixed(10., definition = la.Definition([cells, genes])))

# %%
dist.component_dim

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.value.sum(1))
print(dist.likelihood)

# %% [markdown]
# However, we can manually choose the dependent dimension as well:

# %%
dist = la.distributions.Dirichlet(concentration = la.Fixed(10., definition = la.Definition([cells, genes])), component_dim = cells)

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.value.sum(0))
print(dist.likelihood)

# %% [markdown]
# We can also still add other dependent dimensions

# %%
dist = la.distributions.Dirichlet(concentration = la.Fixed(10., definition = la.Definition([cells, genes])), component_dim = cells, dependent_dims = {genes})

# %%
dist.reset_recursive()
dist.run_recursive()
print(dist.value)
print(dist.value.sum(0))
print(dist.likelihood)

# %% [markdown]
# ## Transformed distributions

# %% [markdown]
# We can directly transform the output of a distribution, using invertible transforms

# %%
dist_1 = la.distributions.Normal(scale = 1., transforms = [la.transforms.Affine(a = 2.)])
dist_2 = la.distributions.Normal(scale = 2.)

# %%
dist_1.reset_recursive()
torch.manual_seed(0)
dist_1.run_recursive()

dist_2.reset_recursive()
torch.manual_seed(0)
dist_2.run_recursive()

# %%
dist_1.likelihood

# %%
dist_1.log_prob(dist_1.value)

# %%
dist_2.log_prob(dist_1.value)

# %%
assert dist_1.likelihood == dist_1.log_prob(dist_1.value)
assert dist_1.likelihood == dist_2.log_prob(dist_1.value)

# %% [markdown]
# ----------

# %%
dist_1 = la.distributions.LogNormal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])), dependent_dims = {cells})
dist_2 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])), dependent_dims = {cells}, transforms = [la.transforms.Exp()])

# %%
dist_1.transforms

# %%
dist_1.reset_recursive()
dist_1.run_recursive()

dist_2.reset_recursive()
dist_2.run_recursive()

# %%
dist_1.likelihood

# %%
dist_2.likelihood

# %% [markdown]
# -----

# %%
dist_1 = la.distributions.LogitNormal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])), dependent_dims = {cells})
dist_2 = la.distributions.Normal(loc = la.Fixed(0., definition = la.Definition([cells])), scale = la.Fixed(2., definition = la.Definition([genes])), dependent_dims = {cells}, transforms = [la.transforms.Logistic()])

# %%
dist_1.transforms

# %%
dist_1.reset_recursive()
dist_1.run_recursive()

dist_2.reset_recursive()
dist_2.run_recursive()

# %%
dist_1.likelihood

# %%
dist_2.log_prob(dist_1.value)

# %% [markdown]
# ------------

# %%
axes = la.Dim(["x", "y"], id = "axis")

# %%
dist_1 = la.distributions.Normal(
    loc = la.Fixed(1., definition = la.Definition([cells, axes])),
    scale = la.Fixed(0.1, definition = la.Definition([genes])),
    transforms = [la.transforms.Circular()],
    dependent_dims = {genes}
)

# %%
dist_1

# %%
dist_1.reset_recursive()
dist_1.run_recursive()

# %%
dist_1.value

# %%
dist_1.likelihood

# %% [markdown]
# ## Reparameterization

# %% [markdown]
# If a distribution is not reparameterizable, the validation will check whether any upstream components are optimizable.

# %% [markdown]
# ## Redefine

# %% [markdown]
# Distributions are used to sample or to calculate log probabilities. If a distribution is a component, it is often redefined to match the desired definition:

# %%
p = la.distributions.Normal()
observation = la.Observation(
    value = pd.Series(0., index = genes.index),
    p = p
)
observation.p

# %%
original_p = la.distributions.Normal()
original_q = la.distributions.Normal(1.)

# %%
latent = la.Latent(
    q = original_q,
    p = original_p,
    definition = la.Definition([genes])
)
print(latent.p)
print(latent.q)

# %% [markdown]
# Be aware that after redefinition, the distribution is not the one you provided:

# %%
assert original_p != latent.p

# %% [markdown]
# However, the upstream components will be still be the same:

# %%
assert original_p.loc == latent.p.loc
