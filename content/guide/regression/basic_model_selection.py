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
n_genes = 100
n_knots = 10
genes = la.Dim([str(i) for i in range(n_genes)], name="gene")
knots = la.Dim(range(n_knots), name="knot")

steps = (
    np.random.choice([-1, 1], (n_genes))[:, None]
    * np.abs(np.random.normal(3.0, 1.0, (n_genes, n_knots)))
    * (np.random.random((n_genes, n_knots)) > 0.5)
)
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
y = la.links.scalar.MonotonicSpline(x=x, a=a, b=intercept)


# %%
# LINEAR GS
# n_genes = 100
# genes = la.Dim([str(i) for i in range(n_genes)], name="gene")

# a_value = (
#     np.random.choice([-1, 1], (n_genes))
#     * (np.random.random((n_genes)) > 0.5)
# )
# a = la.Fixed(pd.Series(a_value, index=genes.index), label="a")
# intercept = la.Fixed(
#     pd.Series(
#         np.random.choice([-1, 1], n_genes)
#         * np.random.normal(1., 1.0, n_genes)
#         * (np.random.random(n_genes) > 0.5),
#         index=genes.index,
#     ),
#     label="intercept",
# )
# y = la.links.scalar.Linear(x=x, a=a, b=intercept)

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
s = la.Parameter(1., transforms = [la.transforms.Exp()])

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
s = la.Parameter(1., transforms = [la.transforms.Exp()])

z = la.links.scalar.Spline(
    x, b=True, output=y.value_definition, label = "z"
)
# z.a.p.step.scale = la.Fixed(1.)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_spline = la.Root(observation = observation)
model_spline.plot()

# %%
s = la.Parameter(1., transforms = [la.transforms.Exp()])
# s = la.Fixed(1.)

z = la.links.scalar.Linear(
    x, a=True, b=True, output=y.value_definition, label = "z"
)
dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_linear = la.Root(observation = observation)
model_linear.plot()

# %%
inference = la.infer.svi.SVI(
    model_mspline, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)
trace.plot()

# %%
inference = la.infer.svi.SVI(
    model_linear, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)
trace.plot()


# %%
inference = la.infer.svi.SVI(
    model_spline, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)
trace.plot()

# %%
model_linear.reset()
a = model_linear.find("z").a
a.run()
print(a.likelihood[0])
print(a.posterior[0])
print(a.posterior[0] - a.likelihood[0])
print(a.likelihood.sum() - a.posterior.sum())
print("--")

a = model_linear.find("z").b
a.run()
print(a.likelihood[0])
print(a.posterior[0])
print(a.posterior[0] - a.likelihood[0])
print(a.likelihood.sum() - a.posterior.sum())
print("--")

obs = model_linear.find("observation")
obs.run()
print(a.likelihood.sum())

# %%
model_spline.reset()
a = model_spline.find("z").a
a.run()
print(a.likelihood[0])
print(a.posterior[0])
print(a.posterior[0] - a.likelihood[0])
print(a.likelihood.sum() - a.posterior.sum())
print(a.likelihood.sum())
print("--")

a = model_spline.find("z").b
a.run()
print(a.likelihood[0])
print(a.posterior[0])
print(a.posterior[0] - a.likelihood[0])
print(a.likelihood.sum() - a.posterior.sum())
print("--")

obs = model_spline.find("observation")
obs.run()

# %%
model_mspline.reset()
a = model_mspline.find("z").a
a.reset()
a.run()
print(a.likelihood[0])
print(a.posterior[0])
print(a.posterior[0] - a.likelihood[0])
print(a.likelihood.sum() - a.posterior.sum())
print(a.likelihood.sum())
print("--")

a = model_mspline.find("z").b
a.run()
print(a.likelihood[0])
print(a.posterior[0])
print(a.posterior[0] - a.likelihood[0])
print(a.likelihood.sum() - a.posterior.sum())
print("--")

obs = model_mspline.find("observation")
obs.run()
print(a.likelihood.sum())

# %%
observed_mspline = la.posterior.vector.VectorObserved(
    model_mspline.observation, retain_samples=model_mspline.components_upstream().values()
)
observed_mspline.sample(30, subsample_n=1)

observed_linear = la.posterior.vector.VectorObserved(
    model_linear.observation, retain_samples=model_linear.components_upstream().values()
)
observed_linear.sample(30, subsample_n=1)

observed_spline = la.posterior.vector.VectorObserved(
    model_spline.observation, retain_samples=model_spline.components_upstream().values()
)
observed_spline.sample(30, subsample_n=1)

# %%
observed_linear.likelihood.item(), observed_spline.likelihood.item(), observed_mspline.likelihood.item()

# %%
observed_linear.elbo.item(), observed_spline.elbo.item(), observed_mspline.elbo.item()

# %%
observed_linear.likelihood.item() - observed_linear.elbo.item(), observed_spline.likelihood.item() - observed_spline.elbo.item(), observed_mspline.likelihood.item() - observed_mspline.elbo.item()

# %%
data = pd.DataFrame({"linear":observed_linear.elbo_features.to_pandas(), "spline":observed_spline.elbo_features.to_pandas(), "mspline": observed_mspline.elbo_features.to_pandas()})
a = "linear";b = "spline"
# a = "spline";b = "mspline"
# a = "mspline";b = "spline"

data["diff"] = data[b] - data[a]
sns.scatterplot(x = a, y = b, hue = "diff", data = data)

# %%
data["diff"].sort_values().plot()

# %%
x_causal_mspline = la.posterior.scalar.ScalarVectorCausal(model_mspline.find("x"), model_mspline.find("observation"))
x_causal_mspline.observed.sample()
x_causal_mspline.sample(10)
x_causal_mspline.sample_empirical()
x_causal_mspline.plot_features()

# %%
x_causal_linear = la.posterior.scalar.ScalarVectorCausal(model_linear.find("x"), model_linear.find("observation"))
x_causal_linear.observed.sample()
x_causal_linear.sample(10)
x_causal_linear.sample_empirical()
x_causal_linear.plot_features()

# %%
x_causal_spline = la.posterior.scalar.ScalarVectorCausal(model_spline.find("x"), model_spline.find("observation"))
x_causal_spline.observed.sample()
x_causal_spline.sample(10)
x_causal_spline.sample_empirical()
x_causal_spline.plot_features()

# %%
data.loc[gene_ids]

# %%
gene_ids = [data["diff"].idxmax(), data["diff"].idxmin()]

x_causal_linear.plot_features(feature_ids = gene_ids, show = True)
x_causal_spline.plot_features(feature_ids = gene_ids, show = True)
x_causal_mspline.plot_features(feature_ids = gene_ids, show = True)
;

# %%

# %%
