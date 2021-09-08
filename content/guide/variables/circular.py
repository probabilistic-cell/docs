# %% [markdown]
# # Circular variables

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
import math

import latenta as la

la.logger.setLevel("INFO")

# %% [markdown]
# ## Transformation of a circular variable
# %% [markdown]
# A circular variable is defined as a variable that "wraps around". To model this in an unconstrained space, we can convert this variable to two orthogonal cartesian coordinates.

# %%
n_samples = 150
sample_ids = [str(i) for i in range(n_samples)]
sample_index = pd.Series(sample_ids, name="cell")

angle_value = pd.Series(
    np.linspace(0.0, (2.0 * np.pi), n_samples) % (2 * math.pi), index=sample_index
)
angle = la.Fixed(
    angle_value,
    distribution=la.distributions.CircularUniform(),
    label="angle",
    symbol=r"\theta",
)
angle


# %%
cartesian = la.links.scalar.Cartesian(angle, label="cartesian")
cartesian


# %%
angle2 = la.links.vector.Circular(cartesian, label="the same angle", symbol=r"\theta 2")
angle2

# %%
angle2.plot()


# %%
assert torch.allclose(angle.prior(), angle2.prior())

# %% [markdown]
# ## Transforming a normal distribution to sample circular values

# %%
coordinates = la.Fixed(
    pd.DataFrame(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        index=pd.Series([0, "0bis", 90, 180, 270], name="angles"),
        columns=pd.Series(["x", "y"], name="axis"),
    )
)
dist1_untransformed = la.distributions.Normal(coordinates, scale=0.5)
dist1 = la.links.vector.Circular(dist1_untransformed)

# %%
posterior = la.posterior.Posterior(dist1)
posterior.sample(200)

posterior.samples[dist1].to_pandas().plot(kind="hist", bins=20, alpha=0.5)

# %% [markdown]
# Correction of the likelihood
# In this case, because the transformation is not bijective, the inverse likelihoods are not corrected.

# %%
dist1.reset()
dist1.run()
dist1.x.value = coordinates.prior()
uncorrected_likelihood = dist1.x.likelihood.sum(-1)
corrected_likelihood = dist1.likelihood

print(corrected_likelihood - uncorrected_likelihood)

# %% [markdown]
# Alternative, assigning the transformation directly to the variable

# %%
dist2 = la.distributions.Normal(
    coordinates, scale=0.5, transforms=[la.transforms.Circular()]
)

# %% [markdown]
# ## Generative model

# %%
n_genes = 100
gene_ids = [str(i) for i in range(n_genes)]
genes = la.Dim(pd.Series(gene_ids, name="gene"))

beta = la.Fixed(
    pd.Series(np.random.uniform(-3, 3, n_genes), index=genes.index), label="beta"
)
shift = la.Fixed(
    pd.Series(np.random.uniform(0.0, 2 * np.pi, n_genes), index=genes.index),
    label="shift",
)
scale = la.Fixed(
    pd.Series(np.random.uniform(0.2, 2.0, n_genes), index=genes.index), label="scale"
)


# %%
y = la.links.scalar.Sine(x=angle, a=beta, shift=shift)


# %%
dist = la.distributions.Normal(loc=y, scale=scale)


# %%
dist.plot()


# %%
sns.lineplot(x=angle.prior().numpy(), y=y.prior()[:, 0].numpy())
sns.scatterplot(x=angle.prior().numpy(), y=dist.prior()[:, 0].numpy())


# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
sns.heatmap(observation_value)

# %% [markdown]
# ## Bayesian modelling

# %%
x = la.Latent(la.distributions.CircularUniform(), definition=angle)


# %%
x.plot()


# %%
output_definition = la.Definition.from_xr(observation_value)
output_definition


# %%
s_dist = la.distributions.LogNormal(1.0, 0.5)
s = la.Latent(s_dist, definition=la.Definition(), label="scale")


# %%
z = la.links.scalar.Sine(
    x,
    a=la.Latent(la.distributions.Normal(), definition=beta),
    shift=la.Latent(la.distributions.CircularUniform(), definition=shift),
    output=output_definition,
    b=True,
)


# %%
z = la.links.scalar.CircularSpline(x, output=output_definition, b=True)


# %%
z.empirical = xr.DataArray(observation_value)


# %%
dist2 = la.distributions.Normal(loc=z, scale=s)

observation = la.Observation(observation_value, dist2, label="observation")


# %%
observation.plot()


# %%
model = la.Root(observation)


# %%
inference = la.infer.svi.SVI(
    model,
    [la.infer.loss.ELBO()],
    la.infer.optim.Adam(lr=0.01),
    subsamplers={"cell": la.infer.subsampling.Subsampler(50)},
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(3000)


# %%
observed = la.posterior.Observed(observation)
observed.sample(10)


# %%
sns.heatmap(observation_value)


# %%
modelled_value = observed.samples[observation.p].mean("sample").to_pandas()
sns.heatmap(modelled_value)


# %%
sns.scatterplot(
    x=angle.prior_pd(),
    y=la.domains.circular.mean(observed.samples[x], "sample").to_pandas(),
)


# %%
assert (
    abs(
        la.math.circular.circular_correlation(
            la.math.circular_mean(observed.samples[x], "sample").to_pandas().values,
            angle.prior_pd().values,
        )
    )
    > 0.8
)

# %%
sns.scatterplot(x=beta.prior_pd(), y=observed.samples[z.b].mean("sample").to_pandas())


# %%
causal = la.posterior.scalar.ScalarVectorCausal(x, observation)
causal.observed.sample(1)
causal.sample(20)
causal.sample_bootstrap(10)
causal.plot_features()


# %% [markdown]
# ## Amortization

# %%
encoder = la.amortization.Encoder(la.Fixed(observation_value, label="observed"), x)


# %%
observation.plot()


# %%
model = la.Root(observation)


# %%
inference = la.infer.svi.SVI(
    model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(10000)


# %%
posterior = la.posterior.Posterior(observation)
posterior.sample(1)


# %%
sns.scatterplot(x=angle.prior_pd(), y=posterior.samples[x].mean("sample").to_pandas())


# %%
assert (
    abs(
        la.math.circular.circular_correlation(
            la.math.circular_mean(observed.samples[x], "sample").to_pandas().values,
            angle.prior_pd().values,
        )
    )
    > 0.8
)

# %%
causal = la.posterior.scalar.ScalarVectorCausal(x, observation)
causal.observed.sample(1)
causal.sample(100)
causal.sample_bootstrap(10)
causal.plot_features(interpretable=observation.p.loc)


# %%
