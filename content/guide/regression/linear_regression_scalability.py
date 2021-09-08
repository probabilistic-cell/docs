# %% [markdown]
# # Linear regression

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
# ## Generative model

# %%
n_cells = 2000
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim(pd.Series(cell_ids, name="cell"))

x = la.Fixed(pd.Series(np.random.uniform(0, 3, n_cells), index=cells.index), label="x")


# %%
n_genes = 1000
genes = la.Dim([str(i) for i in range(n_genes)], id="gene")

slope = la.Fixed(
    pd.Series(
        np.random.choice([-1, 1], n_genes)
        * np.random.normal(3.0, 1.0, n_genes)
        * (np.random.random(n_genes) > 0.5),
        index=genes.index,
    ),
    label="slope",
)
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
y = la.links.scalar.Linear(x=x, a=slope, b=intercept)
dist = la.distributions.Normal(loc=y, scale=scale)


# %%
model_gs = la.Root(dist, label="ground truth", symbol="gs")
model_gs.plot()

# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1, subsample_n=100)


# %%
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)

# %% [markdown]
# ## Linear regression with maximum likelihood

# %%
a = la.Parameter(
    0.0, definition=slope, transforms=la.distributions.Normal(scale=1.0).biject_to()
)
b = la.Parameter(
    0.0, definition=intercept, transforms=la.distributions.Normal(scale=1.0).biject_to()
)
s = la.Parameter(
    1.0, definition=scale, transforms=la.distributions.Exponential().biject_to()
)

z = la.links.scalar.Linear(x, a, b)

dist = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist, label="observation")

# %%
model = la.Root(observation)
model.plot()


# %%
inference = la.infer.svi.SVI(
    model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)
trace.plot()


# %%
observed = la.posterior.Posterior(observation)
observed.sample(10, subsample_n=1)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)
modelled_value = observed.samples[observation.p].sel(sample=0).to_pandas()
sns.heatmap(modelled_value.loc[cell_order], ax=ax1)


# %%
parameter_values = la.qa.cookbooks.check_parameters(
    la.qa.cookbooks.gather_parameters(["intercept", "slope", "x"], model_gs, observed)
)

# %%
causal = la.posterior.scalar.ScalarVectorCausal(x, observation, observed=observed)
causal.sample(10, samples=la.posterior.FileSamples("./test6.nc"))
causal.sample_random(10)
causal.sample_empirical(samples=la.posterior.FileSamples("./test7.nc"))
causal.plot_features(observation.p.loc)

# %%
