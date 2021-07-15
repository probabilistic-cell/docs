# %%
from IPython import get_ipython

# %% [markdown]
# # Mixture distributions

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


# %% [markdown]
# ## Generative model

# %%
n_cells = 500
cell_ids = [str(i) for i in range(n_cells)]

cells = la.Dim(pd.Series(cell_ids, name = "cell"))

x_value = pd.Series(np.random.uniform(0, 5, n_cells), index = cells.index)
x = la.Fixed(x_value, label = "x")


# %%
distributions = {
    "a":la.distributions.Normal(5., 1., definition = la.Definition([cells])),
    "b":la.distributions.Laplace(-1., 1., definition = la.Definition([cells])),
    "c":la.distributions.Normal(10., 1., definition = la.Definition([cells]))
}
components = la.Dim(list(distributions.keys()), "component")


# %%
w = pd.DataFrame(np.vstack([
    x_value * 2,
    np.sqrt(x_value),
    2 * (x_value - 3) ** 2
]), index = components.index, columns = cells.index).T
w = w / w.sum(1).values[:, None]
w = la.Fixed(w)


# %%
mixture = la.distributions.Mixture(weights = w, **distributions, label = "mixture")


# %%
mixture.plot()


# %%
posterior = la.posterior.Posterior(mixture)
posterior.sample(1)


# %%
observation_value = posterior.samples[mixture].sel(sample = 0).to_pandas()
sns.scatterplot(x = x_value, y = observation_value)
sns.kdeplot(x = x_value, y = observation_value)

# %% [markdown]
# ## Modelling the mixture weights with given x

# %%
coefficients = la.links.scalar.Spline(x, output = w, label = "coefficients", transforms = [la.transforms.Softmax(components)], output_distribution = la.distributions.Normal())
dist = la.distributions.Mixture(weights = coefficients, **distributions)
observation = la.Observation(observation_value, dist, label = "observation")
observation.plot()


# %%
inference = la.infer.svi.SVI(observation, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(1000)


# %%
posterior = la.posterior.scalar.ScalarObserved(observation)
posterior.sample(10)


# %%
assert posterior.likelihood / n_cells > -3


# %%
sns.scatterplot(x = x_value, y = observation_value)
sns.kdeplot(x = x_value, y = observation_value)

modelled_value = posterior.samples[dist].stack(cells=("sample", "cell")).to_pandas()
modelled_x = posterior.samples[x].stack(cells=("sample", "cell")).to_pandas()
# sns.scatterplot(x = modelled_x, y = modelled_value)
sns.kdeplot(x = modelled_x, y = modelled_value)


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x, observation)
causal.sample(100)
causal.observed.sample()

causal.plot_features();

# %% [markdown]
# ## Modelling the mixture weights with latent x

# %%
a = la.Latent(la.distributions.Uniform(0., 5.), definition = x)


# %%
coefficients = la.links.scalar.Spline(a, output = w, label = "coefficients", transforms = [la.transforms.Softmax(components)], output_distribution = la.distributions.Normal())


# %%
dist = la.distributions.Mixture(weights = coefficients, **distributions)


# %%
observation = la.Observation(observation_value, dist, label = "observation")


# %%
observation.plot()


# %%
inference = la.infer.svi.SVI(observation, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
trainer = la.infer.trainer.Trainer(inference)


# %%
trace = trainer.train(10000)


# %%
posterior = la.posterior.Posterior(observation)
posterior.sample(10)


# %%
modelled_value = posterior.samples[dist].stack(cells=("sample", "cell")).to_pandas()
modelled_x = posterior.samples[a].stack(cells=("sample", "cell")).to_pandas()
sns.kdeplot(x = modelled_x, y = modelled_value)


# %%
causal = la.posterior.scalar.ScalarVectorCausal(a, observation)
causal.sample(100)
causal.observed.sample()

causal.plot_features();


# %%



