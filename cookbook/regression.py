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
y = la.links.scalar.Linear(x = x, a = slope, b = intercept, scale_x = True)


# %%
dist = la.distributions.Normal(loc = y, scale = scale)

# %%
dist.plot()


# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(observation_value.iloc[np.argsort(x.prior_pd()).values])

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
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(1000)
trace.plot();


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n = 3)


# %%
sns.heatmap(observation_value)


# %%
modelled_value = observed.samples[dist].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value)


# %%
slope_actual = slope.prior_pd()
slope_inferred = observed.samples[a].mean("sample").to_pandas()
sns.scatterplot(x = slope_actual, y = slope_inferred)
assert np.corrcoef(slope_actual, slope_inferred)[0, 1] > 0.8

# %%
intercept_actual = intercept.prior_pd()
intercept_inferred = observed.samples[b].mean("sample").to_pandas()
sns.scatterplot(x = intercept_actual, y = intercept_inferred)
assert np.corrcoef(intercept_actual, intercept_inferred)[0, 1] > 0.8


# %% [markdown]
# ## Linear regression with maximal likelihood and latent x

# %%
la.distributions.Uniform(0., 1.).biject_to()

# %%
x2 = la.Parameter(0.5, definition = x, transforms = la.distributions.Uniform(0., 1.).biject_to(), label = "x")
x2.distribution = la.distributions.Uniform(0., 1.)
a = la.Parameter(0., definition = slope, transforms = la.distributions.Normal(scale = 1.).biject_to())
b = la.Parameter(0., definition = intercept, transforms = la.distributions.Normal(scale = 1.).biject_to())
s = la.Parameter(1., definition = scale, transforms = la.distributions.Exponential().biject_to())

z = la.links.scalar.Linear(x2, a, b)

dist2 = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist2, label = "observation")

# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(1000)


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10)


# %%
sns.heatmap(observation_value)


# %%
modelled_value = observed.samples[dist2].sel(sample = 0).to_pandas()
sns.heatmap(modelled_value)


# %%
x_actual = x.prior_pd()
x_inferred = observed.samples[x2].mean("sample").to_pandas()
sns.scatterplot(x = x_actual, y = x_inferred)
assert np.corrcoef(x_actual, x_inferred)[0, 1] > 0.8

# %%
slope_actual = slope.prior_pd()
slope_inferred = observed.samples[a].mean("sample").to_pandas()
sns.scatterplot(x = slope_actual, y = slope_inferred)
assert np.corrcoef(slope_actual, slope_inferred)[0, 1] > 0.8


# %%
intercept_actual = intercept.prior_pd()
intercept_inferred = observed.samples[b].mean("sample").to_pandas()
sns.scatterplot(x = intercept_actual, y = intercept_inferred)
assert np.corrcoef(intercept_actual, intercept_inferred)[0, 1] > 0.8


# %% [markdown]
# ## Linear regression with variational inference and latent x

# %%
x_dist = la.distributions.Uniform(0., 3.)
x2 = la.Latent(x_dist, definition = x, initial = x.prior_pd())

a_dist = la.distributions.Normal(0., 10., definition = slope.clean)
a = la.Latent(a_dist, definition = slope)

s_dist = la.distributions.LogNormal(0.5, 0.5)
s = la.Latent(s_dist, label = "scale")

z = la.links.scalar.Linear(x = x2, a = a, b = True, scale_x = True, output = observation.clean)

dist = la.distributions.Normal(loc = z, scale = s)

observation = la.Observation(observation_value, dist, label = "observed")


# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(1000)


# %%
observed = la.posterior.vector.VectorObserved(observation)
observed.sample(10)


# %%
x_actual = x.prior_pd()
x_inferred = observed.samples[x2].mean("sample").to_pandas()
sns.scatterplot(x = x_actual, y = x_inferred)
assert np.corrcoef(x_actual, x_inferred)[0, 1] > 0.8

# %%
slope_actual = slope.prior_pd()
slope_inferred = observed.samples[a].mean("sample").to_pandas()
sns.scatterplot(x = slope_actual, y = slope_inferred)
assert np.corrcoef(slope_actual, slope_inferred)[0, 1] > 0.8

# %%
intercept_actual = intercept.prior_pd()
intercept_inferred = observed.samples[z.b].mean("sample").to_pandas()
sns.scatterplot(x = intercept_actual, y = intercept_inferred)
assert np.corrcoef(intercept_actual, intercept_inferred)[0, 1] > 0.8


# %%
z.empirical = xr.DataArray(observation_value)


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x2, observation, observed = observed)
causal.sample(10)
causal.sample_random(10)
causal.plot_features();

# %% [markdown]
# ## Linear regression with variational inference and amortized latent x

# %%
amortization_input = la.Fixed(observation_value, label = "observed")


# %%
nn = la.amortization.Encoder(amortization_input, x2)

# %%
model = la.Model(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(3000)


# %%
observed = la.posterior.vector.VectorObserved(observation)
observed.sample(10)

# %%
x_actual = x.prior_pd()
x_inferred = observed.samples[x2].mean("sample").to_pandas()
sns.scatterplot(x = x_actual, y = x_inferred)
assert np.corrcoef(x_actual, x_inferred)[0, 1] > 0.8

# %%
slope_actual = slope.prior_pd()
slope_inferred = observed.samples[a].mean("sample").to_pandas()
sns.scatterplot(x = slope_actual, y = slope_inferred)
assert np.corrcoef(slope_actual, slope_inferred)[0, 1] > 0.8

# %%
intercept_actual = intercept.prior_pd()
intercept_inferred = observed.samples[z.b].mean("sample").to_pandas()
sns.scatterplot(x = intercept_actual, y = intercept_inferred)
assert np.corrcoef(intercept_actual, intercept_inferred)[0, 1] > 0.8


# %%
z.empirical = xr.DataArray(observation_value)


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x2, observation, observed = observed)
causal.sample(10)
causal.sample_random(1)
causal.plot_features();

# %%
