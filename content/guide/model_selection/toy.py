# %% [markdown]
# # Model selection

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
n_cells = 1000
cells = la.Dim([str(i) for i in range(n_cells)], "cell")

x1 = la.Fixed(
    pd.Series(np.random.uniform(0, 1, cells.size), index=cells.index),
    label="x1",
    symbol="x1",
)
x1.distribution = la.distributions.Uniform(definition=x1)
x2 = la.Fixed(
    pd.Series(np.random.uniform(0, 1, cells.size), index=cells.index),
    label="x2",
    symbol="x2",
)
x2.distribution = la.distributions.Uniform(definition=x2)


# %%
gene_infos = {}
gene_outputs = {}


# %%
n_genes = 20
genes = la.Dim(pd.Series([f"constant {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": "constant"} for gene_id in genes.index})


# %%
def random_a(n_genes):
    return np.random.choice([-1, 1], n_genes) * np.random.uniform(1.0, 2.0, n_genes)


def random_x(n_genes):
    return np.random.uniform(0.15, 0.85, n_genes)


# %%
n_genes = 20
gene_type = "linear(x1)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")

gene_outputs[gene_type] = la.links.scalar.Linear(x1, a)


# %%
n_genes = 20
gene_type = "switch(x1)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")
switch = la.Fixed(pd.Series(random_x(n_genes), index=genes.index))

gene_outputs[gene_type] = la.links.scalar.Switch(x1, a=a, switch=switch)


# %%
n_genes = 20
gene_type = "linear(x2)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")

gene_outputs[gene_type] = la.links.scalar.Linear(x2, a)


# %%
n_genes = 20
gene_type = "spline(x1)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

knot = la.Fixed(
    pd.Series(np.linspace(0.0, 1.0, 10), index=pd.Series(range(10), name="knot"))
)
a = la.Fixed(
    pd.DataFrame(
        np.random.rand(n_genes, knot[0].size) * 2 - 1,
        columns=pd.Series(range(10), name="knot"),
        index=genes.index,
    )
)

gene_outputs[gene_type] = la.links.scalar.Spline(
    x1, a=a, knot=knot, smoothness=la.Fixed(10.0)
)


# %%
n_genes = 20
gene_type = "linear(x1) + linear(x2)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a1 = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")
a2 = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")
gene_outputs[gene_type] = la.modular.Additive(
    x1=la.links.scalar.Linear(x1, a1),
    x2=la.links.scalar.Linear(x2, a2),
    definition=la.Definition([cells, genes]),
)


# %%
n_genes = 20
gene_type = "linear([x1, x2])"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")

gene_outputs[gene_type] = la.links.scalars.Linear([x1, x2], a=a)


# %%
n_genes = 20
gene_type = "switch([x1, x2])"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")
gene_outputs[gene_type] = la.links.scalars.Linear(
    [
        la.links.scalar.Switch(
            x1, switch=la.Fixed(pd.Series(random_x(n_genes), index=genes.index))
        ),
        la.links.scalar.Switch(
            x2, switch=la.Fixed(pd.Series(random_x(n_genes), index=genes.index))
        ),
    ],
    a=a,
)


# %%
n_genes = 20
gene_type = "linear([switch(x1), x2])"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene"))
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index=genes.index), label="a")
gene_outputs[gene_type] = la.links.scalars.Linear(
    [
        la.links.scalar.Switch(
            x1, a=la.Fixed(1.0), switch=la.Fixed(0.5), label="X_activity"
        ),
        x2,
    ],
    a=a,
)


# %%
n_genes = 20
gene_type = "complicated([x1, x2])"
genes.index = pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name="gene")
gene_infos.update({gene_id: {"type": gene_type} for gene_id in genes.index})

n_knots = [10, 10]
knots_value = np.dstack(
    np.meshgrid(np.linspace(0, 1, n_knots[0]), np.linspace(0, 1, n_knots[1]))
).reshape(-1, 2)
knots_dim = la.Dim(pd.Series(range(knots_value.shape[0]), name="knot"))
knots = [
    la.Fixed(pd.Series(knots_value[:, 0], index=knots_dim.index)),
    la.Fixed(pd.Series(knots_value[:, 1], index=knots_dim.index)),
]
a = la.Fixed(
    pd.DataFrame(
        np.random.rand(n_genes, knots_dim.size) * 4 - 1,
        columns=knots_dim.index,
        index=genes.index,
    )
)
smoothness = la.Fixed(10.0)
gene_outputs[gene_type] = la.links.scalars.Thinplate(
    [x1, x2], knots=knots, a=a, smoothnesses=smoothness, n_knots=n_knots
)


# %%
gene_info = pd.DataFrame.from_dict(gene_infos, orient="index")
gene_info.index.name = "gene"


# %%
genes = la.Dim(gene_info.index)
output = la.modular.Additive(
    0.0, la.Definition([cells, genes]), label="output", subsettable={genes}
)


# %%
for component_id, component in gene_outputs.items():
    setattr(output, component_id, component)


# %%
scale = la.Fixed(0.5)


# %%
dist = la.distributions.Normal(loc=output, scale=scale)


# %%
model_gs = la.Root(dist)
model_gs.plot()


# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
cell_order = model_gs.find("x1").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)


# %%
output.empirical = observation_value


# %%
x1_causal = la.posterior.scalar.ScalarVectorConditional(x1, dist, observed=posterior)
x1_causal.sample(5)
x2_causal = la.posterior.scalar.ScalarVectorConditional(x2, dist, observed=posterior)
x2_causal.sample(5)
x1_x2_causal = la.posterior.scalarscalar.ScalarScalarVectorConditional(x1_causal, x2_causal)
x1_x2_causal.sample(5, n_batch=40)


# %%
gene_ids = gene_info.groupby("type").sample(1).index


# %%
x1_causal.plot_features(feature_ids=gene_ids)
x2_causal.plot_features(feature_ids=gene_ids)
x1_x2_causal.plot_features_contour(feature_ids=gene_ids)


# %% [markdown]
# ## Creating the different models

# %%
models = {}

# %% [markdown]
# ### Constant

# %%
output.reset()

# %%
mu = la.modular.Additive(
    intercept=la.Parameter(0.0, la.Definition([genes])),
    definition=output.value_definition,
    subsettable={genes},
)
s = la.Parameter(
    1.0,
    definition=la.Definition([genes]),
    transforms=la.distributions.Exponential().transform_to(),
)


# %%
mu.x1 = la.links.scalar.Constant(x1, output=mu.value_definition)
mu.x2 = la.links.scalar.Constant(x2, output=mu.value_definition)


# %%
dist = la.distributions.Normal(mu, s)
observation = la.Observation(observation_value, dist, label="observation")
mu.empirical = observation_value


# %%
model_constant = la.Root(observation)
models["constant"] = model_constant
model_constant.plot()


# %%
model_constant2 = model_constant.clone()
assert (
    model_constant.observation.p.loc.x1.x.loader.value
    is model_constant2.observation.p.loc.x1.x.loader.value
)
assert not (model_constant.observation.p.scale is model_constant2.observation.p.scale)

# %% [markdown]
# ### Additive spline model

# %%
model = la.Root(model_constant.observation.clone())

mu = model.observation.p.loc
mu.x1 = la.links.scalar.Spline(x1, output=mu)
mu.x2 = la.links.scalar.Spline(x2, output=mu)

models["spline(x1) + spline(x2)"] = model

model.plot()

# %% [markdown]
# ### Thin plate

# %%
model = la.Root(model_constant.observation.clone())

mu = model.observation.p.loc
del mu.x1
del mu.x2
mu.x12 = la.links.scalars.Thinplate({"x1": x1, "x2": x2}, output=mu)

models["thinplate([x1, x2])"] = model
model.plot()

# %% [markdown]
# ### Additive switch model

# %%
model = la.Root(model_constant.observation.clone())
x1 = model.find("x1")
x2 = model.find("x2")

mu = model.observation.p.loc
mu.x1 = la.links.scalar.Switch(x1, shift=True, a=True, output=mu.value_definition)
mu.x2 = la.links.scalar.Switch(x2, shift=True, a=True, output=mu.value_definition)

models["switch(x1) + switch(x2)"] = model
model.plot()

# %% [markdown]
# ### Multiplicative switch model

# %%
model = la.Root(model_constant.observation.clone())
x1 = model.find("x1")
x2 = model.find("x2")

mu = model.observation.p.loc
del mu.x1
del mu.x2
mu.x12 = la.links.scalars.Linear(
    [
        la.links.scalar.Switch(x1, shift=True, output=mu),
        la.links.scalar.Switch(x2, shift=True, output=mu),
    ],
    a=la.Definition([genes]),
)

models["switch([x1, x2])"] = model
model.plot()

# %% [markdown]
# ### Single and multiplicative switch model with shared switch

# %%
model = la.Root(model_constant.observation.clone())
x1 = model.find("x1")
x2 = model.find("x2")

mu = model.observation.p.loc
mu.x1 = la.links.scalar.Switch(x1, shift=True, a=True, output=mu)
mu.x2 = la.links.scalar.Switch(x2, shift=True, a=True, output=mu)
mu.x12 = la.links.scalars.Linear(
    [
        la.links.scalar.Switch(x1, switch=mu.x1.switch),
        la.links.scalar.Switch(x2, switch=mu.x2.switch),
    ],
    a=True,
    output=mu,
)

models["switch(x1, s1) + switch(x2, s2) + switch([x1, x2], [s1, s2])"] = model
model.plot()

# %% [markdown]
# ### Linear additive

# %%
model = la.Root(model_constant.observation.clone())
x1 = model.find("x1")
x2 = model.find("x2")

mu = model.observation.p.loc
mu.x1 = la.links.scalar.Linear(x1, a=True, output=mu)
mu.x2 = la.links.scalar.Linear(x2, a=True, output=mu)

models["linear(x1) + linear(x2)"] = model
model.plot()

# %% [markdown]
# ### Additive linear and interaction

# %%
model = la.Root(model_constant.observation.clone())
x1 = model.find("x1")
x2 = model.find("x2")

mu = model.observation.p.loc
mu.x1 = la.links.scalar.Linear(x1, a=True, output=mu)
mu.x2 = la.links.scalar.Linear(x2, a=True, output=mu)
mu.x12 = la.links.scalars.Linear([x1, x2], a=True, output=mu)

models["linear(x1) + linear(x2) + linear([x1, x2])"] = model
model.plot()

# %% [markdown]
# ### linear(switch(x1), y)

# %%
model = la.Root(model_constant.observation.clone())
x1 = model.find("x1")
x2 = model.find("x2")

mu = model.observation.p.loc
del mu.x1
del mu.x2
mu.x12 = la.links.scalars.Linear(
    [
        la.links.scalar.Switch(
            x1, shift=True, label="x1 activity", symbol="x1_activity", output=mu
        ),
        x2,
    ],
    a=la.Definition([genes]),
    output=mu,
)

models["linear([switch(x1), x2])"] = model
model.plot()

# %% [markdown]
# ### embedding
# %% [markdown]
# To illustrate why model selection is necessary, we can also create an extremely flexible embedding model that will certainly overfit. Overfitting in this context means that the embedding will contain information coming from technical noise.
#
# Note that, in a typical use case, we would use an amortization function to infer the latent space, but this is not necessary in this case as we're simply using a linear function on a simple dataset.

# %%
model = la.Root(model_constant.observation.clone())

mu = model.observation.p.loc
# del mu.x1
# del mu.x2

components = la.Dim(20, "component")
embedding = la.Latent(
    la.distributions.Normal(0.0, 1.0),
    definition=la.Definition([cells, components]),
    label="embedding",
    symbol="embedding",
)
a = la.Latent(la.distributions.Normal(), definition=la.Definition([genes, components]))

mu.embedding = la.links.vector.Matmul(embedding, a)

models["embedding"] = model
model.plot()

# %% [markdown]
# ---------------------

# %%
for model_ix, (model_id, model) in enumerate(models.items()):
    print(model_id)
    #     if model_id != "constant":
    #         continue

    x1 = model.observation.find("x1")
    x2 = model.observation.find("x2")

    #     if "trace" in model: del model["trace"]
    if "trace" not in model:
        inference = la.infer.svi.SVI(
            model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
        )
        trainer = la.infer.trainer.Trainer(inference)
        model["trace"] = trainer.train(1000)

        inference = la.infer.svi.SVI(
            model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
        )
        trainer = la.infer.trainer.Trainer(inference)
        model["trace"] = trainer.train(1000)

    #     if "observed" in model: del model["observed"]
    if "observed" not in model:
        model["observed"] = la.posterior.vector.VectorPredictive(model.observation)
        model["observed"].sample(5)

    #     if "x1_causal" in model: del model["x1_causal"]
    if "x1_causal" not in model:
        x1_causal = la.posterior.scalar.ScalarVectorConditional(
            x1, model.observation, observed=model["observed"]
        )
        x1_causal.sample(20)
        x1_causal.sample_random(5)
        model["x1_causal"] = x1_causal

    #     if "x2_causal" in model: del model["x2_causal"]
    if "x2_causal" not in model:
        x2_causal = la.posterior.scalar.ScalarVectorConditional(
            x2, model.observation, observed=model["observed"]
        )
        x2_causal.sample(20)
        x2_causal.sample_random(5)
        model["x2_causal"] = x2_causal

    #     if "x1_x2_causal" in model: del model["x1_x2_causal"]
    if "x1_x2_causal" not in model:
        x1_x2_causal = la.posterior.scalarscalar.ScalarScalarVectorConditional(
            model["x1_causal"], model["x2_causal"]
        )
        x1_x2_causal.sample(20, n_batch=20)
        model["x1_x2_causal"] = x1_x2_causal

# %% [markdown]
# Explore a model

# %%
# model = models["linear(x1) + linear(x2) + linear([x1, x2])"]
# model = models["thinplate([x1, x2])"]
# model = models["constant"]
# model = models["switch(x1) + switch(x2)"]
# model = models["switch([x1, x2])"]
# model = models["spline(x1) + spline(x2)"]
model = models["linear([switch(x1), x2])"]
# model = models["embedding"]


# %%
model["x1_causal"].plot_features()


# %%
model["x2_causal"].plot_features()


# %%
model["x1_x2_causal"].plot_features_contour(feature_ids=gene_ids)


# %%
model["x1_x2_causal"].plot_features_contour(
    feature_ids=gene_info.query("type == 'linear([switch(x1), x2])'").index[:5]
)

# %%
fig = model["x1_x2_causal"].plot_likelihood_ratio()
fig.axes[0].legend(bbox_to_anchor=(0.5, 1.1), ncol=2, title="coregulatory")


# %%
ax = sns.scatterplot(
    x="lr_x1", y="lr_x2", data=model["x1_x2_causal"].scores.join(gene_info), hue="type"
)
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_xscale("sigmoid")
ax.set_yscale("sigmoid")

# %%
model["x1_x2_causal"].plot_features_contour(
    #     feature_ids = gene_info.query("type == 'switch(x1)'").index
)

# %% [markdown]
# ## Model selection

# %%
model = models["switch(x1) + switch(x2)"]
model = models["switch([x1, x2])"]

# %%
# model = models["thinplate([x1, x2])"]
# model["observed"].sample(1)

# %%
for model_id, model in models.items():
    print(model_id, model["observed"].elbo.mean().item())


# %%
likelihoods = xr.concat(
    [model["observed"].likelihood_features for model in models.values()],
    dim=pd.Series(models.keys(), name="model"),
).to_pandas()
model_ids = [
    model_id for model_id in likelihoods.index if model_id not in ["embedding"]
]
likelihoods = likelihoods.loc[model_ids]
evidences = xr.concat(
    [model["observed"].elbo_features for model in models.values()],
    dim=pd.Series(models.keys(), name="model"),
).to_pandas()
evidences = evidences.loc[model_ids]


# %%
sns.heatmap((likelihoods == likelihoods.max(0)).T)


# %%
sns.heatmap((likelihoods - likelihoods.max(0)).T)


# %%
sns.heatmap((likelihoods.max(0) - likelihoods).T < 5)


# %%
# evidences = evidences.loc[[i for i in evidences.index if not i.startswith("thinpl")]]

# %%
sns.heatmap((evidences).T)


# %%
sns.heatmap((evidences.max(0) - evidences).T < 5)

# %%
sns.heatmap((evidences == evidences.max()).T)


# %%
selected_evidence = evidences == evidences.max()


# %%
likelihood_diff = likelihoods - (likelihoods * selected_evidence).min()
undermodelled = (likelihood_diff > 1).any()


# %%
gene_info["undermodelled"] = undermodelled


# %%
gene_info["selected_evidence"] = selected_evidence.idxmax(0)


# %%
gene_info.groupby(["type", "undermodelled"]).count()[
    "selected_evidence"
].unstack().T.fillna(0.0).style.background_gradient(
    cmap=sns.color_palette("Blues", as_cmap=True)
)


# %%
pd.crosstab(
    gene_info["selected_evidence"], gene_info["type"]
).style.background_gradient(cmap=sns.color_palette("Blues", as_cmap=True))


# %%
sns.stripplot(likelihood_diff.max(), y=gene_info["type"])


# %%
