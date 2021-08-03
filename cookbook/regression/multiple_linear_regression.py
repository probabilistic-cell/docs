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
# # Linear regression with multiple inputs

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
n_cells = 300
cells = la.Dim(pd.Series([str(i) for i in range(n_cells)]), id = "cell")

x1 = la.Fixed(pd.Series(np.random.uniform(0, 3, n_cells), index = cells.index), label = "x1", symbol = "x1")
x2 = la.Fixed(pd.Series(np.random.uniform(0, 3, n_cells), index = cells.index), label = "x2", symbol = "x2")


# %%
n_genes = 100
genes = la.Dim(pd.Series([str(i) for i in range(n_genes)]), id = "gene")

def random_coefficient(n_genes):
    return np.random.choice([-1, 1], n_genes) * np.random.normal(3., 1., n_genes) * (np.random.random(n_genes) > 0.5)

slope_x1 = la.Fixed(pd.Series(random_coefficient(n_genes), index = genes.index), label = "slope_x1")
slope_x2 = la.Fixed(pd.Series(random_coefficient(n_genes), index = genes.index), label = "slope_x2")
slope_x12 = la.Fixed(pd.Series(random_coefficient(n_genes), index = genes.index), label = "slope_x12")
intercept = la.Fixed(pd.Series(random_coefficient(n_genes), index = genes.index), label = "intercept")
scale = la.Fixed(pd.Series(np.random.uniform(1., 1.2, n_genes), index = genes.index), label = "scale")

gene_info = pd.DataFrame({
    "slope_x1" : slope_x1.prior_pd(),
    "slope_x2" : slope_x2.prior_pd(),
    "slope_x12" : slope_x12.prior_pd(),
    "intercept" : intercept.prior_pd(),
})


# %%
z = la.modular.Additive(definition = la.Definition([cells, genes]))

assert all(not la.variables.is_dim_broadcasted(dim) for dim in z)

z.intercept = intercept

assert not la.variables.is_dim_broadcasted(z[0])
assert la.variables.is_dim_broadcasted(z[1])

z.x1 = la.links.scalar.Linear(x = x1, a = slope_x1)
z.x2 = la.links.scalar.Linear(x = x2, a = slope_x2)
z.x12 = la.links.scalars.Linear([x1, x2], a = slope_x12)

assert all(la.variables.is_dim_broadcasted(dim) for dim in z)

# %% [markdown]
# alternative

# %%
z = la.modular.Additive(
    intercept = intercept,
    x1_effect = la.links.scalar.Linear(x = x1, a = slope_x1),
    x2_effect = la.links.scalar.Linear(x = x2, a = slope_x2),
    x12_effect = la.links.scalars.Linear([x1, x2], a = slope_x12),
    definition = la.Definition([cells, genes])
)

assert all(la.variables.is_dim_broadcasted(dim) for dim in z)

# %%
dist = la.distributions.Normal(loc = z, scale = scale)

# %%
model_gs = la.Model(dist, label = "ground truth", symbol = "gs")
model_gs.plot()

# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample = 0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x1").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)

# %% [markdown]
# ## Linear regression with maximum likelihood

# %%
slope_x1 = la.Parameter(0., definition = la.Definition([genes]), label = "slope_x1")
slope_x2 = la.Parameter(0., definition = la.Definition([genes]), label = "slope_x2")
slope_x12 = la.Parameter(0., definition = la.Definition([genes]), label = "slope_x12")
z = la.modular.Additive(
    intercept = intercept,
    x1_effect = la.links.scalar.Linear(x = x1, a = slope_x1),
    x2_effect = la.links.scalar.Linear(x = x2, a = slope_x2),
    x12_effect = la.links.scalars.Linear([x1, x2], a = slope_x12),
    definition = la.Definition([cells, genes])
)


# %%
scale = la.Parameter(1., definition = scale, transforms = [la.transforms.Exp()])
dist = la.distributions.Normal(loc = z, scale = scale)

observation = la.Observation(observation_value, dist, label = "observation")

# %%
model = la.Model(observation, label = "maximum likelihood", symbol = "ml")
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
cell_order = model_gs.find_recursive("x1").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[observation.p].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)


# %%
parameter_ids = ["intercept", "slope_x1", "slope_x2", "slope_x12"]
parameter_values = []
for parameter_id in parameter_ids:
    parameter_values.append(pd.DataFrame({
        "actual":model_gs.find_recursive(parameter_id).prior_pd(),
        "inferred":observed.samples[model.find_recursive(parameter_id)].mean("sample").to_pandas(),
        "parameter_id":parameter_id
    }))
parameter_values = pd.concat(parameter_values)

# %%
fig, axes = la.plotting.axes_wrap(len(parameter_ids))
for ax, parameter_id in zip(axes, parameter_ids):
    parameter_value = parameter_values.query("parameter_id == @parameter_id")
    ax.scatter(x = parameter_value["actual"], y = parameter_value["inferred"])
    ax.set_title(parameter_id)
    assert np.corrcoef(parameter_value["actual"], parameter_value["inferred"])[0, 1] > 0.8, paremter_id

# %% [markdown]
# ## Linear regression with maximal likelihood and parameterized x

# %%
x1 = la.Parameter(0.5, definition = x1, transforms = [la.transforms.Logistic()], label = "x1")

# %%
slope_x1 = la.Parameter(0., definition = la.Definition([genes]), label = "slope_x1")
slope_x2 = la.Parameter(0., definition = la.Definition([genes]), label = "slope_x2")
slope_x12 = la.Parameter(0., definition = la.Definition([genes]), label = "slope_x12")
z = la.modular.Additive(
    intercept = intercept,
    x1_effect = la.links.scalar.Linear(x = x1, a = slope_x1),
    x2_effect = la.links.scalar.Linear(x = x2, a = slope_x2),
    x12_effect = la.links.scalars.Linear([x1, x2], a = slope_x12),
    definition = la.Definition([cells, genes])
)

# %%
scale = la.Parameter(1., definition = scale, transforms = [la.transforms.Exp()])
dist = la.distributions.Normal(loc = z, scale = scale)

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
observed.sample(10)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x1").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[observation.p].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)


# %%
la.qa.cookbooks.check_parameters(la.qa.cookbooks.gather_parameters(["intercept", "slope_x1", "slope_x2", "slope_x12", "x1"], model_gs, observed))

# %% [markdown]
# ## Linear regression with variational inference and latent x

# %%
x1 = la.Latent(la.distributions.Uniform(0., 3.), definition = la.Definition([cells]), label = "x1", symbol = "x1")

# %%
slope_x1_p = la.distributions.Normal(loc = 0., scale = la.Parameter(1., transforms = [la.transforms.Exp()]))
slope_x2_p = la.distributions.Normal(loc = 0., scale = la.Parameter(1., transforms = [la.transforms.Exp()]))
slope_x12_p = la.distributions.Normal(loc = 0., scale = la.Parameter(1., transforms = [la.transforms.Exp()]))

slope_x1 = la.Latent(slope_x1_p, definition = la.Definition([genes]), label = "slope_x1")
slope_x2 = la.Latent(slope_x2_p, definition = la.Definition([genes]), label = "slope_x2")
slope_x12 = la.Latent(slope_x12_p, definition = la.Definition([genes]), label = "slope_x12")

z = la.modular.Additive(
    intercept = intercept,
    x1_effect = la.links.scalar.Linear(x = x1, a = slope_x1),
    x2_effect = la.links.scalar.Linear(x = x2, a = slope_x2),
    x12_effect = la.links.scalars.Linear([x1, x2], a = slope_x12),
    definition = la.Definition([cells, genes])
)

# %%
scale = la.Parameter(1., definition = scale, transforms = [la.transforms.Exp()])
dist = la.distributions.Normal(loc = z, scale = scale)

observation = la.Observation(observation_value, dist, label = "observation")


# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.05))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(5000)
trace.plot();


# %%
observed = la.posterior.vector.VectorObserved(observation)
observed.sample(10)


# %%
la.qa.cookbooks.check_parameters(la.qa.cookbooks.gather_parameters(["intercept", "slope_x1", "slope_x2", "slope_x12", "x1"], model_gs, observed))

# %%
z.empirical = xr.DataArray(observation_value)
causal_x2 = la.posterior.scalar.ScalarVectorCausal(x2, observation, observed = observed)
causal_x2.sample(10)
causal_x2.sample_random(10)
causal_x2.plot_features();

# %%
causal_x1 = la.posterior.scalar.ScalarVectorCausal(x1, observation, observed = observed)
causal_x1.sample(10)
causal_x1.sample_random(10)
causal_x1.plot_features();

# %%
causal_x1_x2 = la.posterior.scalarscalar.ScalarScalarVectorCausal(causal_x1, causal_x2)
causal_x1_x2.sample(10)
causal_x1_x2.plot_likelihood_ratio();

# %%
sns.scatterplot(gene_info["slope_x12"] + gene_info["slope_x2"], causal_x1_x2.scores["lr_x2"], hue = gene_info["slope_x2"])

# %%
causal_x1_x2.plot_features_contour();

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x1").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)
modelled_value = observed.samples[observation.p].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax = ax1)

# %%
