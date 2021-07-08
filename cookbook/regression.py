# %%
from IPython import get_ipython

# %%
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
cell_ids = [str(i) for i in range(n_cells)]
cell_index = pd.Series(cell_ids, name = "cell")

x = la.Fixed(pd.Series(np.random.uniform(0, 3, n_cells), index = cell_index), label = "x")


# %%
n_features = 100
feature_ids = [str(i) for i in range(n_features)]
feature_index = pd.Series(feature_ids, name = "feature")

slope = la.Fixed(pd.Series(np.random.uniform(-3, 3, n_features), index = feature_index), label = "slope")
intercept = la.Fixed(pd.Series(np.random.uniform(-3, 3, n_features), index = feature_index), label = "intercept")
scale = la.Fixed(pd.Series(np.random.uniform(1., 1.2, n_features), index = feature_index), label = "scale")


# %%
y = la.links.scalar.Linear(x, slope, intercept, scale_x = True)


# %%
dist = la.distributions.Normal(loc = y, scale = scale)

# %%
dist.plot()


# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(observation_value.iloc[np.argsort(x.prior_xr()).values])

# %% [markdown]
# ## Maximum likelihood

# %%
x2 = la.Parameter(0.5, definition = x, transforms = la.distributions.Uniform(0., 1.).biject_to(), label = "x")
a = la.Parameter(0., definition = slope, transforms = la.distributions.Normal(scale = 1.).biject_to())
b = la.Parameter(0., definition = intercept, transforms = la.distributions.Normal(scale = 1.).biject_to())
s = la.Parameter(1., definition = scale, transforms = la.distributions.Exponential().biject_to())

z = la.links.scalar.Linear(x2, a, b)

dist2 = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist2, label = "observation")


# %%
observation.plot()

# %%
model = la.Model(observation)


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(1000)


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n = 3)


# %%
sns.heatmap(observation_value)


# %%
modelled_value = observed.samples[dist2].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value)


# %%
sns.scatterplot(
    x = x.prior_xr(),
    y = observed.samples[x2].mean("sample").sel(cell = cell_ids)
)


# %%
sns.scatterplot(
    x = slope.prior_xr(),
    y = observed.samples[a].mean("sample")
)


# %%
sns.scatterplot(
    x = intercept.prior_xr(),
    y = observed.samples[b].mean("sample")
)

# %% [markdown]
# ## Variational inference

# %%
# a_dist = x.distribution.expand(x.dims, x.shape, x.coords)
# a_dist = la.distributions.Normal()
x_dist = la.distributions.Uniform(0., 3.)

x2 = la.Latent(x_dist, definition = x, initial = x.prior_xr())
# a = x


# %%
graph = la.interpretation.ComponentGraph(x2)
graph.display_graphviz(legend = False)


# %%
a_dist = la.distributions.Normal(0., 10., definition = slope)
# b_dist = slope.distribution

a = la.Latent(a_dist, definition = slope)
# b = slope


# %%
graph = la.interpretation.ComponentGraph(a)
graph.display_graphviz(legend = False)


# %%
s_dist = la.distributions.LogNormal(0.5, 0.5)
# s_dist = scale.distribution

s = la.Latent(s_dist, label = "scale")

# s = scale


# %%
graph = la.interpretation.ComponentGraph(s)
graph.display_graphviz(legend = False)


# %%
z = la.links.scalar.Linear(x2, a, b = True, scale_x = True, output = observation.clean)
# z = la.links.scalar.Spline(a, output = observed.clean, n_knots = 10)


# %%
dist2 = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist2, label = "observed")


# %%
graph = la.interpretation.ComponentGraph(observation)
graph.display_graphviz(legend = True)


# %%
model = la.Model(observation)


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(1000)


# %%
observed = la.posterior.vector.VectorObserved(observation)
observed.sample(10)


# %%
sns.heatmap(observation_value)


# %%
modelled_value = observed.samples[dist2].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value)


# %%
sns.scatterplot(
    x = x.prior_xr().to_pandas(),
    y = observed.samples[x2].mean("sample").sel(cell = cell_ids)
)


# %%
sns.scatterplot(
    x = slope.prior_xr().to_pandas(),
    y = observed.samples[a].mean("sample")
)


# %%
z.empirical = xr.DataArray(observation_value)


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x2, observation, observed = observed)
causal.sample(10)
causal.sample_random(10)
causal.plot_features();

# %% [markdown]
# ## Amortization

# %%
amortization_input = la.Fixed(observation_value, label = "observed")


# %%
nn = la.amortization.Encoder(amortization_input, x2)


# %%
observation.plot()


# %%
model = la.Model(observation)


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(3000)


# %%
observed = la.posterior.Observed(observation)
observed.sample(50)


# %%
sns.heatmap(observation_value)


# %%
modelled_value = observed.samples[dist2].mean("sample").to_pandas()
sns.heatmap(modelled_value)


# %%
sns.scatterplot(
    x = x.prior_xr().to_pandas(),
    y = observed.samples[x2].mean("sample")
)


# %%
sns.scatterplot(
    x = slope.prior_xr().to_pandas(),
    y = observed.samples[a].mean("sample")
)


# %%
z.empirical = xr.DataArray(observation_value)


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x2, observation, observed = observed)
causal.sample(10)
causal.plot_features();


# %%



