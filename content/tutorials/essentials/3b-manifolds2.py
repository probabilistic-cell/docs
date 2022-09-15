# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Manifolds

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac
import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables)

# %%
adata = la.data.load_myod1()

# %% tags=["hide-input", "remove-output"]
import scanpy as sc

adata.raw = adata

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

sc.pp.combat(adata)
sc.pp.pca(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])

# %% [markdown]
# ## Embedding: Inferring flexible but uninterpretable latent variables

# %%
adata_oi = adata

# %%
transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata_oi)

# %%
components = la.Dim(20, 
embedding = la.Latent(
    la.distributions.Normal(),
    definition=[transcriptome["cell"]],
    label="differentiation",
)
differentiation

# %% [markdown]
# Now that we have defined the cellular latent space, we still have to define how this space affects the transcriptome. Given that we (can) assume that _most_ genes will change smoothly but non-linearly during differentiation, we choose a spline function for this:

# %%
foldchange = transcriptome.find("foldchange")
foldchange.differentiation = la.links.scalar.Spline(
    differentiation, definition=foldchange.value_definition
)
foldchange.plot()

# %% [markdown]
# :::{note}
#
# Although a spline function is flexible, like always we do still make some assumptions, namely:
# - The outcome is smooth, without a lot of sudden jumps.
# - The flexibility is limited by the number of knots. There is a trade-off here, as more knots will mean more flexibility but also more chances of overfitting.
#
# :::

# %% [markdown]
# As before, we also want to correct for the batch effect (again remember that the foldchange is an additive modular variable, so we can add any components we want and the foldchange will be a linear combination of them):

# %%
batch = la.variables.discrete.DiscreteFixed(adata_oi.obs["batch"])
foldchange.batch = la.links.vector.Matmul(batch, definition=foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

# %% [markdown]
# We can extract the inferred values using a {class}`~latenta.posterior.scalar.ScalarObserved` posterior:

# %%
differentiation_observed = la.posterior.scalar.ScalarObserved(differentiation)
differentiation_observed.sample(10)

# %%
differentiation_observed.plot()

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color=["differentiation", "gene_overexpressed"])

# %% [markdown]
# This may look good at first sight, but not if we take a closer look. After all, about half of the differentiation is used up by cells coming from the control cells!

# %% [markdown]
# ## Differentiation: Providing a little bit of prior knowledge

# %% [markdown]
# In this case, we can easily fix this by including some external information. Namely, we know which cells were not perturbed, and we can therefore _nudge_ the differentiation values of those cells close to 0 by specifying an appropriate prior distribution.

# %% [markdown]
# Before we specified the prior $p$ of the differentiation to be a uniform distribution as we argued that we do not know if our cells are more concentrated at any time point in the differentation process. However, this prior is for now set for all the cells (controls or overexpressing *Myod1*) meaning that all the cells are as likely to contribute to any time of the differentiation process. But this is not true, we know that our control cells are concentrated at the very beggining of the differentiation process as they are undifferentiated. We therefore wants a prior which as the same time is more or less uniform for the *Myod1* cells but close to 0 everywhere except at the very begining for the control cells.

# %%
transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata_oi)

# %%
# gene_overexpressed = la.variables.DiscreteFixed(adata_oi.obs["gene_overexpressed"])

# %% tags=["remove-input", "remove-output"]
# alpha_go = la.Fixed(pd.Series([1., 1.], index = adata_oi.obs["gene_overexpressed"].cat.categories), label = "alpha")
# beta_go = la.Fixed(pd.Series([1., 100.], index = adata_oi.obs["gene_overexpressed"].cat.categories), label = "beta")

# %% [markdown]
# This nudging is performed by an appropriate prior distribution. A beta distribution becomes very handy here, indeed depending on the parameters it can either be a uniform  distribution or can be concentrated at any value we want! So we will now set the prior of the differentiation as a beta distribution for which its two parameters alpha and beta will depend on if it's a control of *Myod1* overexpressing cell.

# %% [markdown]
# Note that a $\beta$(1,1) is a uniform, and a $\beta$(1,100) is very concentrated at 0.

# %%
import pandas as pd

differentiation_p = la.distributions.Beta(
    beta=la.Fixed(
        pd.Series([1.0, 100.0], index=["Myod1", "mCherry"])[
            adata_oi.obs["gene_overexpressed"]
        ].values,
        definition=[transcriptome["cell"]],
    ),
    alpha=la.Fixed(
        pd.Series([1.0, 1.0], index=["Myod1", "mCherry"])[
            adata_oi.obs["gene_overexpressed"]
        ].values,
        definition=[transcriptome["cell"]],
    ),
)

# %% [markdown]
# Note that we do not place a hard prior on these differentiation values, and that control cells can therefore still have high differentiation if this is really supported by the data.

# %%
differentiation = la.Latent(
    differentiation_p, definition=[transcriptome["cell"]], label="differentiation"
)

# %%
differentiation.plot()

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.differentiation = la.links.scalar.Spline(
    differentiation, definition=foldchange.value_definition
)

# %%
batch = la.variables.discrete.DiscreteFixed(adata_oi.obs["batch"])
foldchange.batch = la.links.vector.Matmul(batch, definition=foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

# %%
differentiation_observed = la.posterior.scalar.ScalarObserved(differentiation)
differentiation_observed.sample(10)

# %%
differentiation_observed.plot()

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color=["differentiation", "gene_overexpressed"])

# %%
sc.pl.umap(
    adata_oi, color=adata.var.reset_index().set_index("symbol").loc[["Myog"]].gene
)

# %% [markdown]
# Let's investigate which genes are driven by differentiation:

# %%
differentiation_causal = la.posterior.scalar.ScalarVectorCausal(
    differentiation,
    transcriptome,
    observed=differentiation_observed,
    interpretable=transcriptome.p.mu.expression,
)
differentiation_causal.sample(10)
differentiation_causal.sample_empirical()
differentiation_causal.sample_random()

# %%
differentiation_causal.plot_features()

# %% [markdown]
# ## Differentiation: increasing scalability and robustness through amortization

# %% [markdown]
# Because the model has to infer both cell- and gene-specific latent variables at the same time, finding a robust solution for this model be tricky. One reason for this is that that during optimization, both groups of latent variables have to follow eachother, and they can easily get stuck together in suboptimal solutions. Indeed, if you would run the above model a couple of times, you might notice that sometimes the inferred differentiation values are completely different from the "expected" situation.
#
# One main way to make inference of a latent space more robust is to not directly infer the latent variables for each cell individually, but instead train a _amortization function_ that provides this latent space for us. This function will typically use our observation, in this case the transcriptome's count matrix, and predict the components of the variational distribution, i.e. $\mu$ and $\sigma$. We of course still have to train some parameters, namely the parameters of this amortization function, but this makes training much easier as information is shared between all cells.
#
# The main challenge with amortization is that we need to choose a function that is extremely flexible, as it needs to combine information coming from all genes into a probably highly non-linear model. We therefore typically choose a neural network. The number of layers of this function depends on the complexity of the data. More specifically, how all features in the observation are related. A transcriptomics count matrix has fairly simple "co-expression" structure and therefore typically on requires two layers. The pixels and channels in imaging data on the other hand have a more intracate and hierarchical structure that is best captured by many layers (i.e. deep learning).

# %% [markdown]
# :::{note}
#
# Don't we lose interpretability if we use this neural network?
#
# It's certainly true that a neural network, even of a small size, can be very difficult to interpret. At best, we may be able to rank some genes according to their importance in the model. However, in our case we don't really care about this interpretability, because amortization is just a trick to make inference easier. It's important to understand that the actual interpretability always lies downstream from the variational distribution, in how the cellular latent space is related to the transcriptome. The amortization function just helps us to estimate the variational distribution, but doesn't help us with explaining the transcriptome. For a variational distribution, we therefore only want accuracy and ease of inference, which neural networks can provide, but not interpretability. In fact, in an ideal world, the optimal solution for a model with and without amortization would be the same.
#
# :::

# %%
transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata_oi)

# %%
differentiation = la.Latent(
    differentiation_p, definition=[transcriptome["cell"]], label="differentiation"
)

# %%
encoder = la.amortization.Encoder(
    la.Fixed(transcriptome.loader, definition=transcriptome),
    differentiation,
    pretrain=False,
    lr=1e-3,
)

# %%
differentiation.plot()

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.differentiation = la.links.scalar.Spline(
    differentiation, definition=foldchange.value_definition
)

# %%
batch = la.variables.discrete.DiscreteFixed(adata_oi.obs["batch"])
foldchange.batch = la.links.vector.Matmul(batch, definition=foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=1e-3)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=1e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

# %%
differentiation_observed = la.posterior.scalar.ScalarObserved(differentiation)
differentiation_observed.sample(10)

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color=["differentiation", "batch"])

# %%
differentiation_causal = la.posterior.scalar.ScalarVectorCausal(
    differentiation,
    transcriptome,
    observed=differentiation_observed,
    interpretable=transcriptome.p.mu.expression,
)
differentiation_causal.sample(30)
differentiation_causal.sample_empirical()
differentiation_causal.sample_random()

# %%
differentiation_causal.plot_features()

# %% [markdown]
# Apart from a more robust training, amortization also has a couple of additional advantages:
# - Scalability, because we no longer train the variational distribution's parameters for each individual cell, which can easily go into hundreds of thousands to millions. Instead, we only infer a few dozen to a few hundred hundred parameters of the neural network.
# - We get a function that can predict the latent space even on unseen cells "for free". Note however, that this does not necessarily mean that our function is generalizable, as this may depend on the assumptions, hyperparameters and structure of the neural network.

# %% [markdown]
# ## Cell cycle: Including quite some prior knowledge

# %% [markdown]
# ![](https://cdn.zmescience.com/wp-content/uploads/2018/10/giraffe-661648_960_720.jpg)

# %%
