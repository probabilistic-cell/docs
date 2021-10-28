# %% [markdown]
# # Splines

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
# ## Random walks

# %%
cells = la.Dim(pd.Series(range(100), name="cell").astype(str))
genes = la.Dim(pd.Series(range(4), name="gene").astype(str))

# %%
dist = la.distributions.RandomWalk(10, la.distributions.Normal())

# %%
dist.run()
print(dist.value)
print(dist.likelihood)

# %%
knots = la.Dim(range(10), "knot")

# %%
dist = la.distributions.RandomWalk(
    10, la.distributions.Normal(), definition=la.Definition([genes, knots])
)

# %%
dist.run()
print(dist.value)
print(dist.likelihood)

# %%
knots = la.Dim(range(10), "knot")

# %%
dist = la.distributions.RandomWalk(
    10, la.distributions.Normal(), definition=la.Definition([knots, genes])
)

# %%
dist.reset()
dist.run()
print(dist.value)
print(dist.step.likelihood)
print(dist.likelihood)
print(dist.log_prob(dist.value))
print(dist.log_prob(torch.zeros(dist.value.shape)))

# %%
dist = la.distributions.RandomWalk(
    10, la.distributions.Normal(definition=la.Definition([genes]))
)
dist.plot()

# %%
dist.reset()
dist.run()
print(dist.value)
print(dist.step.likelihood)
print(dist.likelihood)
print(dist.log_prob(dist.value))
print(dist.log_prob(torch.zeros(dist.value.shape)))

# %%
genes = la.Dim(pd.Series(range(100), name="gene").astype(str))

# %%
dist = la.distributions.RandomWalk(
    100, la.distributions.Normal(definition=la.Definition([genes]))
)
dist.run()
value = dist.value.cpu().numpy()
for i in range(value.shape[0]):
    sns.lineplot(x=np.arange(value.shape[1]), y=value[i], alpha=0.3)

# %% [markdown]
# ## Generative model

# %%
n_cells = 50
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(cell_ids, name="cell"))

x = la.Fixed(pd.Series(np.random.uniform(0, 20, n_cells), index=cells.index), label="x")


# %%
n_genes = 100
n_knots = 10
genes = la.Dim([str(i) for i in range(n_genes)], name="gene")
knots = la.Dim(range(n_knots), name="knot")

steps = (
    np.random.choice([-1, 1], (n_genes, n_knots))
    * np.random.normal(3.0, 1.0, (n_genes, n_knots))
    * (np.random.random((n_genes, n_knots)) > 0.5)
)
a_value = steps.cumsum(1)
a_value = a_value - a_value.mean(1, keepdims=True)
a = la.Fixed(pd.DataFrame(a_value, columns=knots.index, index=genes.index), label="a")
intercept = la.Fixed(
    pd.Series(
        np.random.choice([-1, 1], n_genes)
        * np.random.normal(3.0, 1.0, n_genes)
        * (np.random.random(n_genes) > 0.5),
        index=genes.index,
    ),
    label="intercept",
)
scale = la.Fixed(
    pd.Series(np.random.uniform(1.0, 1.2, n_genes), index=genes.index), label="scale"
)


# %%
y = la.links.scalar.Spline(x=x, a=a, b=intercept)
dist = la.distributions.Normal(loc=y, scale=scale, label="distribution")


# %%
model_gs = la.Root(dist = dist, label="ground truth", symbol="gs")
model_gs.plot()

# %%
posterior = la.posterior.Posterior(dist, retain_samples={dist.loc, dist})
posterior.sample(1)


# %%
loc_value = posterior.samples[dist.loc].sel(sample=0).to_pandas()
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)

# %%
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
gene_ids = genes.coords[:10]
fig, axes = la.plotting.axes_wrap(len(gene_ids))
cell_order = model_gs.find("x").prior_pd().sort_values().index
x_value = model_gs.find("x").prior_pd()

for gene_id, ax in zip(gene_ids, axes):
    ax.scatter(x_value.loc[cell_order], observation_value.loc[cell_order, gene_id])
    ax.plot(x_value.loc[cell_order], loc_value.loc[cell_order, gene_id])
# sns.heatmap(observation_value.loc[cell_order], ax = ax0)

# %% [markdown]
# ## Regression with variational inference

# %%
s = la.Parameter(
    1.0, definition=scale, transforms=la.distributions.Exponential().biject_to()
)

z = la.links.scalar.Spline(
    x, b=intercept, knot=model_gs.find("knot"), output=y.value_definition
)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

# %%
model = la.Root(observation = observation)
model.plot()


# %%
inference = la.infer.svi.SVI(
    model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)
trace.plot()
# inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.05))
# trainer = la.infer.trainer.Trainer(inference)
# trace = trainer.train(3000)
# trace.plot();


# %%
observed = la.posterior.Posterior(
    observation, retain_samples=model.components_upstream().values()
)
observed.sample(10, subsample_n=1)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)
modelled_value = observed.samples[observation.p].sel(sample=0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax=ax1)


# %%
x_causal = la.posterior.scalar.ScalarVectorCausal(x, observation)
x_causal.sample(10)

# %%
x_causal.plot_features()

# %%
parameter_values = la.qa.cookbooks.check_parameters(
    la.qa.cookbooks.gather_parameters(["a", "intercept"], model_gs, observed)
)

# %%
dim = la.Dim([1, 2, 3, 4], "h")

# %%
x_value = pd.Series([1.0, 2.0, 3.0, 0.0], index=dim.index, name="x")
x = la.Fixed(x_value)

y = la.links.scalar.Spline(x)

assert y.value_definition[0] == x[0]
assert y.ndim == 1

# %%

# %%
