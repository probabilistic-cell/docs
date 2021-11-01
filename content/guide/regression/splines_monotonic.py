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
cells = la.Dim(pd.Series(range(600), name="cell").astype(str))
genes = la.Dim(pd.Series(range(5), name="gene").astype(str))
knots = la.Dim(range(5), "knot")

# %%
x_value = (
    pd.Series(
        np.random.choice((0, 1, 2, 3, 4), len(cells)), index=cells.index, name="knot"
    )
    .astype(float)
    .astype("category")
)
x = la.variables.discrete.DiscreteFixed(x_value)
a_gs = la.Fixed(pd.Series([0.5, 0, 1, 2, 1.5], index=knots.index).astype(float))
y = la.links.vector.Matmul(x=x, a=a_gs)
obs = la.distributions.Normal(y)

# %%
obs_value = obs.prior_pd()

# %%
a = la.Parameter(0.0, definition=[knots])
y = la.links.vector.Matmul(x=x, a=a)

# %%
dist = la.distributions.Normal(y)
obs = la.Observation(obs_value, dist)

# %%
inference = la.infer.svi.SVI(obs, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)
trace.plot()

# %%
a.prior_pd().plot()

# %%
sns.scatterplot(
    a.prior_pd(), a_gs.prior_pd(), hue=a.prior_pd().index.astype("category")
)

# %%
a_unconstrained = la.Parameter(0.0, definition=[knots])
a = la.links.vector.Monotonic(a_unconstrained)
y = la.links.vector.Matmul(x=x, a=a)

# %%
dist = la.distributions.Normal(y)
obs = la.Observation(obs_value, dist)

# %%
inference = la.infer.svi.SVI(obs, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)
trace.plot()

# %%
torch.minimum(torch.tensor(0.1), torch.tensor([1.0, 2.0, 3.0]))

# %%
a.prior_pd().plot()

# %%
sns.scatterplot(
    a.prior_pd(), a_gs.prior_pd(), hue=a.prior_pd().index.astype("category")
)

# %% [markdown]
# ## Random walks

# %%
sign = la.distributions.Bernouilli(0.5, transforms=[la.transforms.Sign()])
dist = la.distributions.MonotonicRandomWalk(
    11, la.distributions.HalfNormal(), sign=sign
)
knots = dist[0]

# %%
sign.reset()
sign.run()
sign.likelihood

# %%
dist2 = la.distributions.RandomWalk(len(knots), la.distributions.Normal())

# %%
dist.reset()
dist.run()
dist.value
print("lik     : " + str(dist.likelihood))
print("log_prob: " + str(dist._log_prob(dist.value)))
# assert torch.isclose(dist.log_prob(dist.value), dist.likelihood, atol = 1e-3)
""

# %%
dist.value

# %%
import math

dist.reset()
dist.run()
dist2.run()
print(dist2.log_prob(dist.value))
print(dist.log_prob(dist.value))
print(dist.likelihood)

fold_difference = (dist.likelihood - dist2.log_prob(dist.value)) / math.log(2.0)
print(fold_difference)
assert torch.isclose(
    fold_difference, torch.tensor(len(knots)).float() - 1, atol=1e-1, rtol=1e-1
), "A monotonic spline with HalfNormal should have double the likelihood per step as a regular spline with Normal"
""

# %%
steps = torch.cat([torch.tensor([0.0]), torch.ones(len(knots) - 1)])

value = torch.cumsum(steps, 0)
lik = dist.log_prob(value)
print(lik)

#
steps = -torch.cat([torch.tensor([0.0]), torch.ones(len(knots) - 1)])

value = torch.cumsum(steps, 0)
lik = dist.log_prob(value)
print(lik)

# %%
#
steps = torch.cat([torch.tensor([0.0]), torch.ones(len(knots) - 1)])
steps[: int(len(steps) / 2) + 1] *= -1
print(steps)

value = torch.cumsum(steps, 0)
lik = dist.log_prob(value)
print(lik)

# %% [markdown]
# ----

# %%
latent = la.Latent(dist)
# latent.q = la.distributions.Delta(latent.q.loc, transforms = latent.q.transforms, dependent_dims = latent.q.dependent_dims)

# latent.q.scale.loader.value.data[1:] = torch.log(torch.tensor(0.1))

# %%
latent.reset()
latent.run()
print(latent.likelihood)
print(latent.posterior)

# %%
latent.q.prior()

# %%
latent.q.loc = la.links.vector.Monotonic(latent.q.loc)

# %%
latent.plot()

# %%
# latent.q.loc.loader.value.data[0] = 3.

liks = []
for i in range(500):
    latent.reset()
    latent.run()
    liks.append(latent.likelihood)
torch.cat(liks).mean()

# %%
for i in range(50):
    latent.reset()
    latent.run()

    value = latent.p.value.cpu().numpy()
    sns.lineplot(x=np.arange(value.shape[0]), y=value, alpha=0.1, color="red")

    value = latent.q.value.detach().cpu().numpy()
    sns.lineplot(x=np.arange(value.shape[0]), y=value, alpha=0.1, color="blue")

# %%
inference = la.infer.svi.SVI(
    latent, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)
trace.plot()


# %%
latent.q.loc.x.prior_pd().plot()

# %%
for i in range(50):
    latent.reset()
    latent.run()

    value = latent.p.value.cpu().numpy()
    sns.lineplot(x=np.arange(value.shape[0]), y=value, alpha=0.1, color="red")

    value = latent.q.value.detach().cpu().numpy()
    sns.lineplot(x=np.arange(value.shape[0]), y=value, alpha=0.1, color="blue")

# %%
knots = dist[0]

# %%
dist.plot()

# %%
knots = la.Dim(range(10), "knot")

# %%
dist = la.distributions.MonotonicRandomWalk(
    10, la.distributions.HalfNormal(), definition=la.Definition([genes, knots])
)

# %%
dist.plot()

# %%
dist.run()
print(dist.value)
print(dist.likelihood)

# %%
knots = la.Dim(range(10), "knot")

# %%
dist = la.distributions.MonotonicRandomWalk(
    10, la.distributions.HalfNormal(), definition=la.Definition([knots, genes])
)

# %%
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
dist = la.distributions.MonotonicRandomWalk(
    10, la.distributions.HalfNormal(definition=la.Definition([genes]))
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
genes = la.Dim(pd.Series(range(1), name="gene").astype(str))

# %%
dist = la.distributions.MonotonicRandomWalk(
    n_knots=len(knots),
    step=la.distributions.HalfNormal(definition=la.Definition([genes])),
    sign=la.distributions.Bernouilli(0.5, transforms=[la.transforms.Sign()]),
)
dist.run()
value = dist.value.cpu().numpy()
for i in range(value.shape[0]):
    sns.lineplot(x=np.arange(value.shape[1]), y=value[i], alpha=0.3)

# %% [markdown]
# ## Generative model

# %%
n_cells = 100
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(cell_ids, name="cell"))

x = la.Fixed(pd.Series(np.random.uniform(0, 20, n_cells), index=cells.index), label="x")


# %%
n_knots = 10

knots = la.Dim(range(n_knots), name="knot")

steps = np.vstack(
    [
        np.zeros(n_knots),
        1 * np.ones(n_knots),
        np.random.choice([-1, 1])
        * np.abs(np.random.normal(3.0, 1.0, n_knots))
        * (np.arange(n_knots) > (n_knots / 2)),
    ]
)
n_genes = steps.shape[0]
genes = la.Dim([str(i) for i in range(n_genes)], name="gene")

a_value = steps.cumsum(1)
# a_value = a_value - a_value.mean(1, keepdims=True)
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
model_gs = la.Root(dist=dist, label="ground truth", symbol="gs")
model_gs.plot()

# %%
posterior = la.posterior.Posterior(dist, retain_samples={dist.loc, dist})
posterior.sample(1)


# %%
a_value

# %%
loc_value = posterior.samples[dist.loc].sel(sample=0).to_pandas()
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)

# %%
a_value

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
s = la.Parameter(
    1.0, definition=scale, transforms=la.distributions.Exponential().transform_to()
)

z = la.links.scalar.MonotonicSpline(
    x, b=True, knot=model_gs.find("knot"), output=y.value_definition
)
z.a.q.loc = la.links.vector.Monotonic(z.a.q.loc)
# z.a.p.sign.probs = la.Fixed(0.5)
# z.a.p.step.scale = la.Fixed(1.)
# z.b.p.loc = la.Fixed(0.)
# z.b.p.scale = la.Fixed(3.)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_mspline = la.Root(observation=observation)
model_mspline.plot()

# %%
s = la.Parameter(
    1.0, definition=scale, transforms=la.distributions.Exponential().transform_to()
)

z = la.links.scalar.Spline(
    x, b=True, knot=model_gs.find("knot"), output=y.value_definition
)
# z.a.p.step.scale = la.Fixed(1.)
# z.b.p.loc = la.Fixed(0.)
# z.b.p.scale = la.Fixed(3.)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_spline = la.Root(observation=observation)
model_spline.plot()

# %%
s = la.Parameter(
    1.0, definition=scale, transforms=la.distributions.Exponential().transform_to()
)

z = la.links.scalar.Linear(x, a=True, b=True, output=y.value_definition)
# z.a.p.scale = la.Fixed(1.)
# z.b.p.loc = la.Fixed(0.)
# z.b.p.scale = la.Fixed(3.)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

model_linear = la.Root(observation=observation)
model_linear.observation.p.loc.a.label = "a"
model_linear.observation.p.loc.b.label = "b"
model_linear.plot()

# %%
models = {"linear": model_linear, "mspline": model_mspline, "spline": model_spline}

# %%
for model_id, model in models.items():
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
pd.Series(
    {model_id: model["observed"].elbo.item() for model_id, model in models.items()}
)

# %%
for model_id, model in models.items():
    print(model_id)
    model.reset()
    a = model.find("a")
    a.run()
    a.likelihood
    #     print(a.likelihood)
    #     print(a.posterior)
    print(a.likelihood - a.posterior)
#     print(a.q.scale.value)
#     print("--")
#     print(a.q.scale.value)

# %%
likelihood_features = pd.DataFrame(
    {
        model_id: model["observed"].likelihood_features.to_pandas()
        for model_id, model in models.items()
    }
).T
likelihood_features.style.bar(axis=0)

# %%
elbo_features = pd.DataFrame(
    {
        model_id: model["observed"].elbo_features.to_pandas()
        for model_id, model in models.items()
    }
).T
elbo_features.style.bar(axis=0)

# %%
(likelihood_features - elbo_features).style.bar(axis=0)

# %%
kl_a = pd.DataFrame(
    {
        model_id: model["observed"]
        .samples[model.find("a").uuid + "_kl"]
        .mean("sample")
        .to_pandas()
        .sum()
        for model_id, model in models.items()
    }
).T
kl_a.style.bar(axis=0)

# %%
kl_b = pd.DataFrame(
    {
        model_id: model["observed"]
        .samples[model.find("b").uuid + "_kl"]
        .mean("sample")
        .to_pandas()
        for model_id, model in models.items()
    }
).T
kl_b.style.bar(axis=0)

# %%
for model_id, model in models.items():
    causal = model["causal"]
    fig = causal.plot_features(show=True)

# %%

# %%
