# %% [markdown]
# # Monotonic splines

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

# %%
cells = la.Dim(pd.Series(range(100), name="cell").astype(str))
genes = la.Dim(pd.Series(range(4), name="gene").astype(str))
knots = la.Dim(range(10), "knot")

# %% [markdown]
# ## Generative model

# %%
n_cells = 100
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(cell_ids, name="cell"))

x = la.Fixed(pd.Series(np.random.uniform(0, 20, n_cells), index=cells.index), label="x")


# %%
# MONOTONIC GS
n_knots = 10

knots = la.Dim(range(n_knots), name="knot")

n_genes = 4
steps = np.concatenate([
    np.concatenate([np.random.normal(3.0, 1.0, (2))[:, None] * np.ones((2, 1)), -np.zeros((2, n_knots - 1))], 1),
    np.random.choice([-1, 1], (n_genes))[:, None]
    * np.abs(np.random.normal(3.0, 1.0, (n_genes, n_knots)))
    * (np.random.random((n_genes, n_knots)) > 0.5),
    np.concatenate([np.ones((n_genes, int(np.floor(n_knots/2)))), -np.ones((n_genes, int(np.ceil(n_knots/2))))], 1)
    * np.abs(np.random.normal(3.0, 1.0, (n_genes, n_knots)))
    * (np.random.random((n_genes, n_knots)) > 0.5)
], 0)

n_genes = steps.shape[0]
genes = la.Dim([str(i) for i in range(n_genes)], name="gene")

a_value = steps.cumsum(1)
a_value = a_value - a_value.mean(1, keepdims=True)
a = la.Fixed(pd.DataFrame(a_value, columns=knots.index, index=genes.index), label="a")
intercept = la.Fixed(
    pd.Series(
        np.random.choice([-1, 1], n_genes)
        * np.random.normal(1., 1.0, n_genes)
        * (np.random.random(n_genes) > 0.5),
        index=genes.index,
    ),
    label="intercept",
)
y = la.links.scalar.Spline(x=x, a=a, b=intercept)


# %%
scale = la.Fixed(
    1.
#     pd.Series(np.random.uniform(1.0, 1.2, n_genes), index=genes.index), label="scale"
)
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
scale.reset()
intercept.reset()
y.reset()

# %%
s = la.Parameter(1., transforms = [la.transforms.Exp()], definition = [genes])

z = la.links.scalar.MonotonicSpline(
    x, b=True, output=y.value_definition, label = "z"
)
# z.a.p.step.scale = la.Fixed(1.)
# z.a.p.sign.probs = la.Fixed(0.5)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_mspline = la.Root(observation = observation)
model_mspline.plot()

# %%
s = la.Parameter(1., transforms = [la.transforms.Exp()], definition = [genes])

z = la.links.scalar.Spline(
    x, b=True, output=y.value_definition, label = "z"
)
# z.a.p.step.scale = la.Fixed(1.)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_spline = la.Root(observation = observation)
model_spline.plot()

# %%
s = la.Parameter(1., transforms = [la.transforms.Exp()], definition = [genes])
# s = la.Fixed(1.)

z = la.links.scalar.Linear(
    x, a=True, b=True, output=y.value_definition, label = "z"
)
dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_linear = la.Root(observation = observation)
model_linear.plot()

# %%
models = {"linear":model_linear, "mspline":model_mspline, "spline":model_spline}
# models = {"mspline":model_mspline}

# %%
for model_id, model in models.items():
    print(model_id)
    inference = la.infer.svi.SVI(
        model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

# %%
for model_id, model in models.items():
    observed = la.posterior.vector.VectorObserved(
        model.observation, retain_samples=model.components_upstream().values()
    )
    observed.sample(30, subsample_n=1)
    model["observed"] = observed
    
    causal = la.posterior.scalar.ScalarVectorCausal(x, model.observation)
    causal.observed.sample()
    causal.sample(10)
    causal.sample_empirical()
    model["causal"] = causal

# %%
pd.Series({model_id:model["observed"].elbo.item() for model_id, model in models.items()})

# %%
likelihood_features = pd.DataFrame({model_id:model["observed"].likelihood_features.to_pandas() for model_id, model in models.items()})
likelihood_features.head(10).style.bar(axis = 1)

# %%
elbo_features = pd.DataFrame({model_id:model["observed"].elbo_features.to_pandas() for model_id, model in models.items()})
elbo_features.head(10).style.bar(axis = 1)

# %%
(likelihood_features - elbo_features).head(10).style.bar(axis = 1)

# %%
for model_id, model in models.items():
    causal = model["causal"]
    fig = causal.plot_features(show = True)

# %%

# %%
