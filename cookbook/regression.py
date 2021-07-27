# %% [markdown]
# # Regression

# %%
from IPython import get_ipython
if get_ipython():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

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
# ## Generative model

# %%
n_cells = 50
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(cell_ids, name = "cell"))

x = la.Fixed(pd.Series(np.random.uniform(0, 3, n_cells), index = cells.index), label = "x")


# %%
n_genes = 100
genes = la.Dim([str(i) for i in range(n_genes)], id = "gene")

slope = la.Fixed(pd.Series(np.random.choice([-1, 1], n_genes) * np.random.normal(3., 1., n_genes) * (np.random.random(n_genes) > 0.5), index = genes.index), label = "slope")
intercept = la.Fixed(pd.Series(np.random.choice([-1, 1], n_genes) * np.random.normal(3., 1., n_genes) * (np.random.random(n_genes) > 0.5), index = genes.index), label = "intercept")
scale = la.Fixed(pd.Series(np.random.uniform(1., 1.2, n_genes), index = genes.index), label = "scale")


# %%
y = la.links.scalar.Linear(x = x, a = slope, b = intercept)
dist = la.distributions.Normal(loc = y, scale = scale)


# %%
model_gs = la.Model(dist, label = "ground truth", symbol = "gs")
model_gs.plot()

# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample = 0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)

# %% [markdown]
# ## Linear regression with maximum likelihood

# %%
a = la.Parameter(0., definition = slope, transforms = la.distributions.Normal(scale = 1.).biject_to())
b = la.Parameter(0., definition = intercept, transforms = la.distributions.Normal(scale = 1.).biject_to())
s = la.Parameter(1., definition = scale, transforms = la.distributions.Exponential().biject_to())

z = la.links.scalar.Linear(x, a, b)

dist = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist, label = "observation")

# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.05))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)
trace.plot();


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n = 3)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)


# %%
parameter_values = la.qa.cookbooks.check_parameters(la.qa.cookbooks.gather_parameters(["intercept", "slope", "x"], model_gs, observed))

# %% [markdown]
# ## Linear regression with maximal likelihood and inferred x

# %%
x = la.Parameter(0.5, definition = x, transforms = la.distributions.Uniform(0., 1.).biject_to(), label = "x")

# %%
a = la.Parameter(0., definition = slope, transforms = la.distributions.Normal(scale = 1.).biject_to())
b = la.Parameter(0., definition = intercept, transforms = la.distributions.Normal(scale = 1.).biject_to())
s = la.Parameter(1., definition = scale, transforms = la.distributions.Exponential().biject_to())

z = la.links.scalar.Linear(x, a, b)

dist = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist, label = "observation")

# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)
trace.plot();


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n = 3)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)


# %%
parameter_values = la.qa.cookbooks.check_parameters(la.qa.cookbooks.gather_parameters(["intercept", "slope", "x"], model_gs, observed))

# %% [markdown]
# ## Linear regression with variational inference and latent x

# %%
x = la.Latent(la.distributions.Uniform(0., 3.), definition = la.Definition([cells]), label = "x")

z = la.links.scalar.Linear(x, output = la.Definition([cells, genes]), a = True, b = True)

dist = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist, label = "observation")

# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.05))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)
trace.plot();


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n = 3)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)


# %%
parameter_values = la.qa.cookbooks.check_parameters(la.qa.cookbooks.gather_parameters(["intercept", "slope", "x"], model_gs, observed))

# %%
z.empirical = xr.DataArray(observation_value)
x.distribution = la.distributions.Uniform(0., 3.)
causal = la.posterior.scalar.ScalarVectorCausal(x, observation, observed = observed)
causal.sample(10)
causal.sample_random(10)
causal.plot_features(observation.p.loc);

# %% [markdown]
# ## Linear regression with variational inference and amortized latent x

# %%
amortization_input = la.Fixed(observation_value, label = "observed")
nn = la.amortization.Encoder(amortization_input, x)


# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.05))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)
trace.plot();


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n = 3)


# %%
parameter_values = la.qa.cookbooks.check_parameters(la.qa.cookbooks.gather_parameters(["intercept", "slope", "x"], model_gs, observed))

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)

# %%
z.empirical = xr.DataArray(observation_value)
x.distribution = la.distributions.Uniform(0., 3.)
causal = la.posterior.scalar.ScalarVectorCausal(x, observation, observed = observed)
causal.sample(10)
causal.sample_random(10)
causal.plot_features();

# %%
