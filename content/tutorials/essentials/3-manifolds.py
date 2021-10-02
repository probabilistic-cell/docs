# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
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
# We'll use the same dataset as [before](./1-variables)...

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

# %%
sc.pl.umap(adata, color=["gene_overexpressed", "batch", "log_overexpression"])

# %% [markdown]
# So far, we have focused on models that basically look like this:

# %% [markdown]
# ![](_static/previous.svg)

# %% [markdown]
# But what if our cellular variables are also unknown, i.e. latent?

# %% [markdown]
# ![](_static/now.svg)

# %% [markdown]
# In this case, apart from learning a latent space about genes, we _at the same time_ also try to learn a latent space on cells, also known as the "cellular manifold". There are many different ways to represent this manifold, and how to do this depends on what we know about the cells and what we try to accomplish:
#
# - **embedding**: Each cell is placed in an n-dimensional space, with only a few restrictions, such as the cell positions are all centered around 0. This space is very flexible, but not interpretable. Essentially, it is only interpretable when converted to a 2D representation and viewed by the human eye. Related to this, this space is typically far removed from biological reality. A cell's state does not "float" in an n-dimensional space.
# - **scalar** (differentiation, cell cycle, ...): Each cell is placed in a one-dimensional space, with a single pseudotime.

# %% [markdown]
# While many manifold models are relatively easy to implement, the main difficulty lies in the interpretability. Especially when different **cellular processes** are happening at the same time in a cell, a latent variable will try to explain all of them. What is therefore often required is the inclusion of prior knowledge that can help with disentangling different cellular processes.

# %% [markdown]
# ## Differentiation: Inferring a dominant scalar latent variable

# %% [markdown]
# Just by looking at the 2d representation, it already becomes obvious that there are two dominant processes going on in the cell: differentiation (in this case to myocytes) and the cell cycle.

# %%
cellcycle_genes = lac.cell.cellcycle.get_cellcycle_genes()
sc.tl.score_genes_cell_cycle(adata, s_genes = cellcycle_genes.query("phase == 'S'")["gene"].tolist(), g2m_genes = cellcycle_genes.query("phase == 'G2/M'")["gene"].tolist())

# %%
sc.pl.umap(adata_oi, color=["gene_overexpressed", "batch", "log_overexpression", "phase"])
sc.pl.umap(adata_oi, color=adata_oi.var.set_index("symbol").loc[symbols]["ens_id"], title = symbols)

# %% [markdown]
# For now, for illustration purpose, we can simply remove (or reduce) the effect of the cell cycle by removing the cycling cells.

# %%
adata_oi = adata[adata.obs["phase"] == "G1"].copy()

# %%
transcriptome = lac.transcriptome.Transcriptome.from_adata(adata_oi)

# %% [markdown]
# We define the differentiation as a _scalar_ latent variable, that assigns to each cell one value. This single value in our case is again modelled as a Normal distribution that captures its uncertainty.
#
# Crucial here is that we provide an appropriate prior distribution. Given that we assume that differentiation has a start and an end, we want to place the cells somewhere in the  $[0, 1]$ interval. A uniform distribution is there appropriate. Do note that other cellular processes may have other assumptions or hypotheses, and will therefore require different priors as we will see later.

# %%
differentiation = la.Latent(la.distributions.Uniform(), definition = [transcriptome["cell"]], label = "differentiation")

# %% [markdown]
# Now that we have defined the cellular latent space, we still have to define how this space affects the transcriptome. We typically choose a spline function for this, as this is a flexible but smooth function.

# %% [markdown]
# :::{note}
#
# Although a spline function is flexible, we still make a couple of assumptions:
# - The outcome is smooth, without a lot of sudden jumps.
# - The flexibility is limited by the number of knots. There is a trade-off here, as more knots will mean more flexibility but also more chances of overfitting.
#
# :::

# %%
foldchange = transcriptome.find("foldchange")
foldchange.differentiation = la.links.scalar.Spline(differentiation, output = foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %% [markdown]
# To extract the inferred values from the 

# %%
differentiation_observed = la.posterior.scalar.ScalarObserved(differentiation)
differentiation_observed.sample(10)

# %%
differentiation_observed.plot()

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color = ["differentiation"])

# %% [markdown]
# Even though the differentiation is very dominant, the model still used about half of the latent space to explain some heterogeneity in the control cells.

# %% [markdown]
# ## Differentiation: Providing a little bit of prior knowledge

# %% [markdown]
# In this case, we can easily fix this by including some external information. Namely, we know which cells were not perturbed, and we can therefore _nudge_ the differentiation values of those cells close to 0 by specifying an appropriate prior distribution.

# %%
import pandas as pd

# %%
differentiation_p = la.distributions.Beta(
    concentration0 = la.Fixed(pd.Series([1., 100.], index = ["Myod1", 'mCherry'])[adata_oi.obs["gene_overexpressed"]].values, definition = la.Definition([cells])),
    concentration1 = la.Fixed(pd.Series([1., 1.], index = ["Myod1", 'mCherry'])[adata_oi.obs["gene_overexpressed"]].values, definition = la.Definition([cells]))
)

# %% [markdown]
# Note that we do not place a hard prior on these differentiation values, and that control cells can therefore still have high differentiation if this is really supported by the data.

# %%
differentiation = la.Latent(differentiation_p, definition = la.Definition([cells]), label = "differentiation")

# %%
# encoder = la.amortization.Encoder(la.Fixed(transcriptome.loader, definition = transcriptome), differentiation, pretrain = True)

# %%
differentiation.plot()

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.differentiation = la.links.scalar.Spline(differentiation, output = foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.01)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
differentiation_observed = la.posterior.scalar.ScalarObserved(differentiation)
differentiation_observed.sample(10)

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color = ["differentiation", "batch"])

# %%
sc.pl.umap(adata_oi, color = adata.var.reset_index().set_index("symbol").loc[["Myog"]].gene)

# %%
differentiation_causal = la.posterior.scalar.ScalarVectorCausal(
    differentiation, transcriptome, observed = differentiation_observed, interpretable = transcriptome.p.mu.expression
)
differentiation_causal.sample(10)

# %% [markdown]
# This posterior also contains samples, but now for some pseudocells with each a distinct value for the `overexpression` variable.

# %% [markdown]
# Depending on the type of causal posterior, you can plot the outcome. The {class}`~latenta.posterior.scalar.ScalarVectorCausal` can for example plot each individual _feature_ across all cells (in this case gene):

# %%
differentiation_causal.sample_empirical()
differentiation_causal.sample_random()

# %%
differentiation_causal.plot_features();

# %% [markdown]
# ## Differentiation: increasing scalability and robustness through amortization

# %% [markdown]
# Because the model has to infer both cell- and gene-specific latent variables at the same time, finding a robust solution for this model be tricky. One reason for this is that that during optimization, both groups of latent variables have to follow eachother, and they can easily get stuck together in suboptimal solutions. Indeed, if you would run the above model a couple of times, you will see that sometimes the inferred differentiation values are completely different.
#
# One main way to make inference of a latent space more robust is to not directly infer the latent variables for each cell individually, but instead train a _amortization function_ that provides this latent space for us. This function will typically use our observation, in this case the transcriptome's count matrix, and predict the components of the variational distribution, i.e. $\mu$ and $\sigma$. We of course still have to train some parameters, namely the parameters of this amortization function, but this makes training much easier as information is shared between all cells.
#
# The main challenge with amortization is that we need to choose a function that is extremely flexible, as it needs to combine information coming from all genes into a probably highly non-linear model. We therefore typically choose a neural network, with a couple . The number of layers of this function depends on the complexity of the data. A transcriptomics count matrix is fairly simple and typically on requires two layers, while imaging data typically has a more intracate and hierarchical structure that is best captured by many layers (i.e. deep learning).

# %% [markdown]
# :::{note}
#
# Don't we lose interpretability if we use this neural network? It's certainly true that a neural network, even of a small size, can be very difficult to interpret. At best, we may be able to rank some genes according to their importance in the model. However, in our case we don't really care about this interpretability, because amortization is just a trick to make inference easier. It's important to remember that the actual interpretability always lies downstream from the variational distribution, in how the cellular latent space is related to the transcriptome. For a variational distribution, we only want accuracy, which neural networks can provide, but often not interpretability.
#
# :::

# %%
differentiation = la.Latent(differentiation_p, definition = la.Definition([cells]), label = "differentiation")

# %%
transcriptome.reset()

# %%
encoder = la.amortization.Encoder(la.Fixed(transcriptome.loader, definition = transcriptome), differentiation, pretrain = False, lr = 1e-2)

# %%
differentiation.plot()

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.differentiation = la.links.scalar.Spline(differentiation, output = foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch("cuda"):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=1e-4)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
differentiation_observed = la.posterior.scalar.ScalarObserved(differentiation)
differentiation_observed.sample(10)

# %%
adata_oi.obs["differentiation"] = differentiation_observed.mean.to_pandas()

# %%
sc.pl.umap(adata_oi, color = ["differentiation", "batch"])

# %%
differentiation_causal = la.posterior.scalar.ScalarVectorCausal(
    differentiation, transcriptome, observed = differentiation_observed, interpretable = transcriptome.p.mu.expression
)
differentiation_causal.sample(10)

# %%
differentiation_causal.sample_empirical()
differentiation_causal.sample_random()

# %%
differentiation_causal.plot_features();

# %%

# %% [markdown]
# Apart from a more robust training, amortization also has a couple of additional advantages:
# - We are more scalable because we no longer need to train parameters for each individual cell, which can easily go into hundreds of thousands. Instead, we only need to infer a few dozen or hundred parameters of the neural network.
# - For free, we get a function that can tell us the latent space even on unseen cells. Note however, that this does not necessarily mean that our function is really generalizable, as this may depend on the assumptions, hyperparameters and structure of the neural network.

# %% [markdown]
# ## Cell cycle: Including quite some prior knowledge

# %%
