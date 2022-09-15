# %% [markdown]
# ## Classification

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

import latenta as la

la.logger.setLevel("INFO")

# %% [markdown]
# ## Generative model

# %%
n_samples = 100
sample_ids = [str(i) for i in range(n_samples)]
sample_index = pd.Series(sample_ids, name="cell")

x = la.Fixed(
    pd.Series(np.random.uniform(0, 3, n_samples), index=sample_index), label="x"
)


# %%
n_clusters = 4
cluster_ids = [str(i) for i in range(n_clusters)]
clusters = la.Dim(cluster_ids, "feature")

beta = la.Fixed(pd.Series([-2, -2, 2, 2], index=clusters.index), "beta")

bias = la.Fixed(pd.Series([1, 1, -1, -1], clusters.index), label="bias")


# %%
concentration = la.links.scalar.Linear(
    x, beta, b=bias, transforms=[la.transforms.Exp()]
)


# %%
probs = la.distributions.Dirichlet(concentration)


# %%
dist = la.distributions.OneHotCategorical(probs)


# %%
dist.plot()

# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(10)


# %%
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
sns.heatmap(observation_value.iloc[np.argsort(x.prior_pd())])

# %%
root_gs = la.Root(dist = dist)

# %% [markdown]
# ## Classification based on observed x

# %%
cluster_definition = la.Definition.from_xr(observation_value)


# %%
concentration = la.links.scalar.Linear(
    x, b=True, a=True, output=cluster_definition, transforms=[la.transforms.Exp()]
)


# %%
probs = la.distributions.Dirichlet(concentration)
onehot = la.distributions.OneHotCategorical(probs)
observation = la.Observation(observation_value, onehot, label="observation")


# %%
observation.plot()


# %%
root = la.Root(observation = observation)


# %%
inference = la.infer.svi.SVI(
    root, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)


# %%
posterior = la.posterior.Posterior(observation, retain_samples={x, probs})
posterior.sample(10)


# %%
sns.heatmap(observation_value.iloc[np.argsort(x.prior().numpy())])


# %%
modelled_value = posterior.samples[probs].mean("sample").to_pandas()
sns.heatmap(modelled_value.iloc[np.argsort(x.prior().numpy())])


# %%
causal = la.posterior.scalar.ScalarVectorCausal(
    x, observation, observed=posterior, retain_samples={probs.concentration}
)
# causal.observed.sample(1)
causal.sample(100)
causal.plot_features(probs.concentration)

# %% [markdown]
# ## Classification based on observed x (2)
# %% [markdown]
# An example with softmax transform

# %%
probs = la.links.scalar.Linear(
    x, b=True, a=True, output=cluster_definition, transforms=[la.transforms.Softmax()]
)
probs.a.p.scale = la.Fixed(1.)
probs.b.p.scale = la.Fixed(1.)
probs.b.p.loc = la.Fixed(1.)


# %%
onehot = la.distributions.OneHotCategorical(probs)
observation = la.Observation(observation_value, onehot, label="observation")


# %%
observation.plot()


# %%
root = la.Root(observation = observation)


# %%
inference = la.infer.svi.SVI(
    root, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(10000)


# %%
posterior = la.posterior.Posterior(observation, retain_samples={x, probs})
posterior.sample(10)

# %%
sns.heatmap(observation_value.iloc[np.argsort(x.prior().numpy())])


# %%
modelled_value = posterior.samples[probs].mean("sample").to_pandas()
sns.heatmap(modelled_value.iloc[np.argsort(x.prior().numpy())])


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x, observation)
causal.observed.sample(1)
causal.sample(100)
causal.plot_features()

# %% [markdown]
# ## Classification based on latent x (and known coefficients)

# %%
x = la.variables.discrete.Discrete(definition = [la.Dim(sample_index), clusters])

# %%
concentration = la.links.scalar.Linear(
    x, b=root_gs.dist.probs.concentration.b, a=root_gs.dist.probs.concentration.a, output=cluster_definition, transforms=[la.transforms.Exp()]
)


# %%
probs = la.distributions.Dirichlet(concentration)
onehot = la.distributions.OneHotCategorical(probs)
observation = la.Observation(observation_value, onehot, label="observation")


# %%
observation.plot()


# %%
root = la.Root(observation = observation)


# %%
inference = la.infer.svi.SVI(
    root, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)


# %%
posterior = la.posterior.Posterior(root, retain_samples={x, root.observation.p, root.observation.p.probs})
posterior.sample(10)


# %%
sns.heatmap(posterior.samples[x].sel(sample = 0).to_pandas().iloc[np.argsort(root_gs.dist.probs.concentration.x.prior_pd())])

# %%
sns.heatmap(posterior.samples[x].mean("sample").to_pandas().iloc[np.argsort(root_gs.dist.probs.concentration.x.prior_pd())])

# %%
