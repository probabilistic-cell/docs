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

x1 = la.Fixed(pd.Series(np.random.uniform(0, 1, cells.size), index = cells.index), label = "x1", symbol = "x1")
x1.distribution = la.distributions.Uniform()
x2 = la.Fixed(pd.Series(np.random.uniform(0, 1, cells.size), index = cells.index), label = "x2", symbol = "x2")
x2.distribution = la.distributions.Uniform()


# %%
gene_infos = {}
gene_outputs = {}


# %%
n_genes = 20
genes = la.Dim(pd.Series([f"constant {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":"constant"} for gene_id in genes.index})


# %%
def random_a(n_genes):
    return np.random.choice([-1, 1], n_genes) * np.random.uniform(1., 2., n_genes)
def random_x(n_genes):
    return np.random.uniform(0.15, 0.85, n_genes)


# %%
n_genes = 20
gene_type = "linear(x1)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")

gene_outputs[gene_type] = la.links.scalar.Linear(x1, a)


# %%
n_genes = 20
gene_type = "switch(x1)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")
switch = la.Fixed(pd.Series(random_x(n_genes), index = genes.index))

gene_outputs[gene_type] = la.links.scalar.Switch(x1, a = a, switch = switch)


# %%
n_genes = 20
gene_type = "linear(x2)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")

gene_outputs[gene_type] = la.links.scalar.Linear(x2, a)


# %%
n_genes = 20
gene_type = "spline(x1)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

knot = la.Fixed(pd.Series(np.linspace(0., 1., 10), index = pd.Series(range(10), name = "knot")))
a = la.Fixed(pd.DataFrame(np.random.rand(n_genes, knot[0].size) * 2 - 1, columns = pd.Series(range(10), name = "knot"), index = genes.index))

gene_outputs[gene_type] = la.links.scalar.Spline(x1, a = a, knot = knot, smoothness = la.Fixed(10.))


# %%
n_genes = 20
gene_type = "linear(x1) + linear(x2)"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a1 = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")
a2 = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")
gene_outputs[gene_type] = la.modular.Additive(
    x1 = la.links.scalar.Linear(x1, a1),
    x2 = la.links.scalar.Linear(x2, a2),
    definition = la.Definition([cells, genes])
)


# %%
n_genes = 20
gene_type = "linear([x1, x2])"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")

gene_outputs[gene_type] = la.links.scalars.Linear([x1, x2], a = a)


# %%
n_genes = 20
gene_type = "switch([x1, x2])"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")
gene_outputs[gene_type] = la.links.scalars.Linear([
    la.links.scalar.Switch(x1, switch = la.Fixed(pd.Series(random_x(n_genes), index = genes.index))), 
    la.links.scalar.Switch(x2, switch = la.Fixed(pd.Series(random_x(n_genes), index = genes.index)))
], a = a)


# %%
n_genes = 20
gene_type = "linear([switch(x1), x2])"
genes = la.Dim(pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene"))
gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

a = la.Fixed(pd.Series(random_a(n_genes), index = genes.index), label = "a")
gene_outputs[gene_type] = la.links.scalars.Linear([
    la.links.scalar.Switch(x1, a = la.Fixed(1.), switch = la.Fixed(0.5), label = "X_activity"),
    x2
], a = a)


# %%
# n_genes = 20
# gene_type = "complicated([X, Y])"
# genes.index = pd.Series([f"{gene_type} {i}" for i in range(n_genes)], name = "gene")
# gene_infos.update({gene_id:{"type":gene_type} for gene_id in genes.index})

# knot = la.Fixed(pd.DataFrame(np.random.rand(10, 2), index = pd.Series(range(10), name = "knot"), columns = pd.Series(range(2), name = "dimension")))
# a = la.Fixed(pd.DataFrame(np.random.rand(n_genes, knot[0].size) * 2 - 1, columns = pd.Series(range(10), name = "knot"), index = genes.index))
# smoothness = la.Fixed(10.)
# gene_outputs[gene_type] = la.links.scalars.Thinplate(
#     [X, Y], knot = knot, a = a, smoothnesses = smoothness
# )


# %%
gene_info = pd.DataFrame.from_dict(gene_infos, orient = "index")
gene_info.index.name = "gene"


# %%
genes = la.Dim(gene_info.index)
output = la.modular.Additive(0., la.Definition([cells, genes]), label = "output", subsettable = ("gene",))


# %%
for component_id, component in gene_outputs.items():
    setattr(output, component_id, component)


# %%
scale = la.Fixed(1.)


# %%
dist = la.distributions.Normal(loc = output, scale = scale)


# %%
model_gs = la.Model(dist)
model_gs.plot()


# %%
posterior = la.posterior.Posterior(dist)
posterior.sample(1)


# %%
observation_value = posterior.samples[dist].sel(sample = 0).to_pandas()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10, 5))
cell_order = model_gs.find_recursive("x1").prior_pd().sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax = ax0)


# %%
output.empirical = observation_value


# %%
x1_causal = la.posterior.scalar.ScalarVectorCausal(x1, dist, observed = posterior)
x1_causal.sample(5)
x2_causal = la.posterior.scalar.ScalarVectorCausal(x2, dist, observed = posterior)
x2_causal.sample(5)
x1_x2_causal = la.posterior.scalarscalar.ScalarScalarVectorCausal(x1_causal, x2_causal)
x1_x2_causal.sample(5, n_batch = 40)


# %%
gene_ids = gene_info.groupby("type").sample(1).index


# %%
x1_causal.plot_features(feature_ids = gene_ids);
x2_causal.plot_features(feature_ids = gene_ids);
x1_x2_causal.plot_features_contour(feature_ids = gene_ids);


# %% [markdown]
# ## Creating the different models

# %%
models = {}

# %% [markdown]
# ### Constant

# %%
output.reset_recursive()

# %%
mu = la.modular.Additive(intercept = la.Parameter(0., la.Definition([genes])), definition = output.value_definition, subsettable = ("gene",))
# mu.distribution = la.distributions.Normal(scale = 3.)
s = la.Parameter(1., definition = la.Definition([genes]), transforms = la.distributions.Exponential().biject_to())


# %%
mu.x1_effect = la.links.scalar.Constant(x1, output = mu.value_definition)
mu.x2_effect = la.links.scalar.Constant(x2, output = mu.value_definition)


# %%
dist = la.distributions.Normal(mu, s)
observation = la.Observation(observation_value, dist, label = "observation")
mu.empirical = observation_value


# %%
model_constant = la.Model(observation)
models["constant"] = model_constant
model_constant.plot()


# %%
model_constant2 = model_constant.clone()
assert (model_constant.observation.p.loc.x1_effect.x.loader.value is model_constant2.observation.p.loc.x1_effect.x.loader.value)
assert not (model_constant.observation.p.scale is model_constant2.observation.p.scale)

# %% [markdown]
# ### Additive spline model

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc
mu.x1_effect = la.links.scalar.Spline(x1, output = mu.value_definition)
mu.x2_effect = la.links.scalar.Spline(x2, output = mu.value_definition)

models["spline(x1) + spline(x2)"] = model

model.plot()

# %% [markdown]
# ### Thin plate

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc
del mu.X
del mu.Y
mu.XY = la.links.scalars.Thinplate([X, Y], output = mu)

la.interpretation.ComponentGraph(model).display_graphviz()
models["thinplate([X, Y])"] = model

# %% [markdown]
# ### Additive switch model

# %%
model = la.Model(model_constant.observation.clone())
x1 = model.find_recursive("x1")
x2 = model.find_recursive("x2")

mu = model.observation.p.loc
mu.x1_effect = la.links.scalar.Switch(x1, output = mu.value_definition)
mu.x2_effect = la.links.scalar.Switch(x2, output = mu.value_definition)

models["switch(x1) + switch(x2)"] = model
model.plot()

# %% [markdown]
# ### Multiplicative switch model

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc
del mu.X
del mu.Y
mu.XY = la.links.scalars.Switch([X, Y], output = mu)

la.interpretation.ComponentGraph(model).display_graphviz()
models["switch([X, Y])"] = model

# %% [markdown]
# ### Single and multiplicative switch model with shared switch

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc
mu.X = la.links.scalar.Switch(X, output = mu)
mu.Y = la.links.scalar.Switch(Y, output = mu)
mu.XY = la.links.scalars.Switch([X, Y], switches = [mu.X.switch, mu.Y.switch], output = mu)

la.interpretation.ComponentGraph(model).display_graphviz()
models["switch(x, sx) + switch(y, sy) + switch([X, Y], [sx, sy])"] = model

# %% [markdown]
# ### Linear additive

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc
mu.X = la.links.scalar.Linear(X, output = mu)
mu.Y = la.links.scalar.Linear(Y, output = mu)

la.interpretation.ComponentGraph(model).display_graphviz()
models["linear(x1) + linear(x2)"] = model

# %% [markdown]
# ### Additive linear and interaction

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc
mu.X = la.links.scalar.Linear(X, output = mu)
mu.Y = la.links.scalar.Linear(Y, output = mu)
mu.XY = la.links.scalars.Linear([X, Y], output = mu)

la.interpretation.ComponentGraph(model).display_graphviz()
models["linear(x1) + linear(x2) + linear([X, Y])"] = model

# %% [markdown]
# ### linear(switch(x1), y)

# %%
model = la.Model(model_constant.observation.clone())

mu = model.observation.p.loc

del mu.X
del mu.Y
mu.XY = la.links.scalars.Linear(
    [la.links.scalar.Switch(X, a = False, label = "x_activity", output = mu), Y],
    output = mu,
    a = la.Latent(mu.distribution, definition = mu[-1:])
)

la.interpretation.ComponentGraph(model).display_graphviz()
models["linear([switch(x1), y])"] = model

# %% [markdown]
# ### embedding
# %% [markdown]
# To illustrate why model selection is necessary, we can also create an extremely flexible embedding model that will certainly overfit. Overfitting in this context means that the embedding will contain information coming from technical noise.
#
# Note that, in a typical use case, we would use an amortization function to infer the latent space, but this is not necessary in this case as we're simply using a linear function on a simple dataset.

# %%
# if True:
if True:
    model = la.Model(model_constant.observation.clone())

    mu = model.observation.p.loc

    components = la.Dim(20, "component")
    embedding = la.Latent(la.distributions.Normal(0., 1.), definition = la.Definition([cells, components]), event_dims = 1)
    a = la.Latent(la.distributions.Normal(), definition = la.Definition([genes, components]))

    mu.embedding_effect = la.links.vector.Matmul(embedding, a)

    la.interpretation.ComponentGraph(model).display_graphviz()
    models["embedding"] = model

# %% [markdown]
# ---------------------

# %%
for model_ix, (model_id, model) in enumerate(models.items()):
    print(model_id)
#     if model_id != "constant":
#         continue

    x1 = model.observation.find_recursive("x1")
    x2 = model.observation.find_recursive("x2")
    
#     if "trace" in model: del model["trace"]
    if "trace" not in model:
        inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.05))
        trainer = la.infer.trainer.Trainer(inference)
        model["trace"] = trainer.train(1000)
        
        inference = la.infer.svi.SVI(model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr = 0.01))
        trainer = la.infer.trainer.Trainer(inference)
        model["trace"] = trainer.train(1000)
    
#     if "observed" in model: del model["observed"]
    if "observed" not in model:
        model["observed"] = la.posterior.vector.VectorObserved(model.observation)
        model["observed"].sample(5)
    
#     if "x1_causal" in model: del model["x1_causal"]
    if "x1_causal" not in model:
        x1_causal = la.posterior.scalar.ScalarVectorCausal(x1, model.observation, observed = model["observed"])
        x1_causal.sample(20)
        x1_causal.sample_random(5)
        model["x1_causal"] = x1_causal
        
#     if "x2_causal" in model: del model["x2_causal"]
    if "x2_causal" not in model:
        x2_causal = la.posterior.scalar.ScalarVectorCausal(x2, model.observation, observed = model["observed"])
        x2_causal.sample(20)
        x2_causal.sample_random(5)
        model["x2_causal"] = x2_causal
        
#     if "x1_x2_causal" in model: del model["x1_x2_causal"]
    if "x1_x2_causal" not in model:
        x1_x2_causal = la.posterior.scalarscalar.ScalarScalarVectorCausal(model["x1_causal"], model["x2_causal"])
        x1_x2_causal.sample(20, n_batch = 20)
        model["x1_x2_causal"] = x1_x2_causal

# %% [markdown]
# Explore a model

# %%
# model = models["linear(x1) + linear(x2) + linear([X, Y])"]
# model = models["thinplate([X, Y])"]
# model = models["constant"]
model = models["switch(x1) + switch(x2)"]
# model = models["spline(x1) + spline(x2)"]
# model = models["linear([switch(x1), y])"]
# model = models["embedding"]


# %%
model["x1_causal"].plot_features();


# %%
model["x2_causal"].plot_features();


# %%
model["x1_x2_causal"].plot_features_contour(feature_ids = gene_ids);


# %%
fig = model["x1_x2_causal"].plot_likelihood_ratio();
fig.axes[0].legend(bbox_to_anchor=(0.5, 1.1), ncol = 2, title = "coregulatory")
fig.tight_layout()


# %%
ax = sns.scatterplot(x = "lr_x1", y = "lr_x2", data = model["x1_x2_causal"].scores.join(gene_info), hue = "type")
ax.legend(bbox_to_anchor=(1.1, 1.05))
ax.set_xscale("sigmoid")
ax.set_yscale("sigmoid")

# %% [markdown]
# ## Model selection

# %%
for model_id, model in models.items():
    print(model_id, model["observed"].elbo.mean().item())


# %%
likelihoods = xr.concat([model["observed"].likelihood_features for model in models.values()], dim = pd.Series(models.keys(), name = "model")).to_pandas()
evidences = xr.concat([model["observed"].elbo_features for model in models.values()], dim = pd.Series(models.keys(), name = "model")).to_pandas()


# %%
sns.heatmap((likelihoods == likelihoods.max(0)).T)


# %%
sns.heatmap((likelihoods - likelihoods.max(0)).T)


# %%
sns.heatmap((likelihoods.max(0) - likelihoods).T < 5)


# %%
sns.heatmap((evidences).T)


# %%
sns.heatmap((evidences == evidences.max()).T)


# %%
selected_evidence = (evidences == evidences.max())


# %%
likelihood_diff = (likelihoods - (likelihoods * selected_evidence).min())
undermodelled = (likelihood_diff > 1).any()


# %%
gene_info["undermodelled"] = undermodelled


# %%
gene_info["selected_evidence"] = selected_evidence.idxmax(0)


# %%
gene_info.groupby(["type", "undermodelled"]).count()["selected_evidence"].unstack().T.fillna(0.).style.background_gradient(cmap=sns.color_palette("Blues", as_cmap=True))


# %%
pd.crosstab(gene_info["selected_evidence"], gene_info["type"]).style.background_gradient(cmap=sns.color_palette("Blues", as_cmap=True))


# %%
sns.stripplot(likelihood_diff.max(), y = gene_info["type"])


# %%



