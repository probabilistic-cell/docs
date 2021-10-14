# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression

# %%
import latenta as la
import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables)...

# %%
import scanpy as sc

adata = la.data.load_myod1()
adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])
adata.var["label"] = adata.var["symbol"]
adata.raw = adata

# %% [markdown]
#
# ## Easy things first: Linear regression

# %% [markdown]
# In the last tutorial, we learned the main types of variables, and how to connect these to construct a model:
# - You either specify them at initialization: `la.Observation(p = ...)`
# - Or connect them afterwards: `observation.p = ...`
#
# We can now use these concepts to build a basic model of a transcriptome:

# %%
genes = la.Dim(adata.var)
cells = la.Dim(adata.obs)

# %%
overexpression = la.Fixed(
    adata.obs["log_overexpression"], label="overexpression", symbol="overexpression"
)

# %%
expression = la.links.scalar.Linear(
    overexpression,
    a=True,
    b=True,
    label="expression",
    symbol = r"\mu",
    definition=la.Definition([cells, genes]),
    transforms=[la.transforms.Exp()],
)

# %% [markdown]
# Although the latent slope and baseline were created automatically for us by specifying `a = True, b = True`, it's important to remember that we could have easily created these variables ourselves:
#
# ```python
# slope = la.Latent(
#     p = la.distributions.Normal(scale = la.Parameter(1., transforms = [la.transforms.Exp()])),
#     definition = la.Definition([genes])
# )
# ```

# %%
dispersion = la.Latent(
    la.distributions.LogNormal(
        la.Parameter(0.0), la.Parameter(1.0, transforms=[la.transforms.Exp()])
    ),
    definition=la.Definition([genes]),
)

# %%
transcriptome_p = la.distributions.NegativeBinomial2(
    mu=expression, dispersion=dispersion
)

# %%
transcriptome = la.Observation(
    adata.X,
    transcriptome_p,
    definition=la.Definition([cells, genes]),
    label="transcriptome",
)

# %%
transcriptome.plot()

# %% [markdown]
# Note the  number of parameters that form the leaves of our model. Why are there so many even for a simple case of linear regression?
#
# Let's remind ourselves what we're actually trying to accomplish. We're trying to create a good model of our observations. 
#
# It's true that there are many models that will provide a very good fit of the data equally, even simple ones. For example, we could just give the actual count matrix as input to the negative binomial and this trivial model would fit extremely well. However it might overfit and it would not help us to understand/learn from our observations. 
#
# So we don't just want a model. We want a model that can explain our observation well, while being both generalizeable and interpretable. And to accomplish this, we have to limit the flexibility that our model can have. You have already done this by specifying two types of priors:
#
# - Hard priors are those that completely constrain the model. For example, by specifying a linear function we don't allow any non-linearities.
# - Soft priors are those that simply push the latent variables towards more likely values. For example, we want to discourage extreme slopes by specifying its distribution.
#
# All these parameters thus serve two purposes: 
# * the parameters of the variational distributions $q$ will try to explain the observations while also remaining faithful to the prior distributions $p$. 
# * The parameters of $p$ on the other hand will try to accomodate the parameters of $q$ as well as possible, but it cannot do this perfectly as these parameters are shared across all genes. 
#
# It's this pushing and pulling between priors and variational distributions that prevent overfitting and underfitting of the model. At the same time, we get some estimates of the uncertainty of our latent variables for free!

# %% [markdown]
# But how do we infer these parameters in practice? We have to find a solution that best balances the needs of the prior distribution with those of the observations. And one of the fastest ways to do that is to use gradient descent, which starts from an initial value and then tries to move these initial values slowly but surely into values fitting the model better. 
#
# These tasks are fullfilled by:
# * a loss function (`ELBO`), represents the "cost" of our observation. An optimization problem seeks to minimize it. 
# * an optimizer (`Adam`), tries to select the best values with regards to our priors/constrains
# * an overarching training class (`SVI`):

# %% [markdown] tags=["remove-output", "remove-input"]
# I put it in bullet points in case we would like to explain a bit more what each do, but it's maybe unecessary and a bit too much? Anyway my explanations might not be the best :P

# %%
inference = la.infer.svi.SVI(
    transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)

# %% [markdown]
# We can now train for a couple of iterations or until the loss function has reached a plateau (= converged):

# %%
trace = trainer.train(10000)
trace.plot()

# %% [markdown]
# You can check that our values have changed. For example, let's look at the 

# %% [markdown] tags=["remove-output", "remove-input"]
# Changed with the iterations? Maybe in the code would be good to see a "before"/beggining and after?

# %%
transcriptome.p.mu.a.q.loc.run()
transcriptome.p.mu.a.q.loc.value_pd.head()

# %% [markdown]
#
# ## Interpreting a regression model
#
# For interpretation of the model, we can then use 3 main types of posteriors:

# %% [markdown]
# ### Observed posteriors
#
# Because we are working with a probabilistic model, every time we run through the model our results will change. For example, each time we run the variational distribution of the slope $q$ the output will be different, which will affect any downstream function even if they are themselves deterministic. To interpret the results, we can thus sample multiple times from the model:

# %%
transcriptome_observed = la.posterior.Observed(transcriptome, retain_samples = {expression.a})
transcriptome_observed.sample(5)

# %% [markdown]
# `.samples` is a dictionary containing the samples of each variable that was upstream of the transcriptome. We can access each variable either by providing the variable itself:

# %% [markdown] tags=["remove-input", "remove-output"]
# either by ... or by...?

# %%
transcriptome_observed.samples[expression.a]

# %% [markdown]
# :::{note}
# Latenta makes extensive use of the [xarray](https://xarray.pydata.org/en/stable/) library for annotated data in more than 2 dimensions. All samples are always stored as {py:class}`xarray.DataArray` objects. Important functions to know are:
# - {py:meth}`~xarray.DataArray.sel` to select a subset of the data
# - {py:meth}`~xarray.DataArray.to_pandas` to convert a 2D or 1D array to a pandas DataFrame or Series
# - {py:attr}`~xarray.DataArray.dims` to get the dimensions of the data
# - {py:attr}`~xarray.DataArray.values` to get the values as a numpy array
# - {py:meth}`~xarray.DataArray.mean` to get the mean across a dimension
# - {py:class}`xarray.DataArray` to construct a new DataArray
# :::

# %%
transcriptome_observed.samples[expression.a].mean("sample")

# %% [markdown]
# ### Causal posteriors
#
# To know how one variable influences another, we use a causal posterior. In essense, this posterior will set a variable of your choice to particular values, and then see how an output variables (and any intermediates) are affected. Latenta contains many different types of causal posteriors, which mainly differ in their visualization capabilities. Here we will use a {class}`~latenta.posterior.scalar.ScalarVectorCausal` posterior, because we are studying how a **scalar** variable (one value for each cell) impacts a **vector** (gene expression for each cell):

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression, transcriptome
)
overexpression_causal.sample(10)

# %% [markdown]
# This posterior also contains samples, but now for some pseudocells with each a distinct value for the `overexpression` variable.

# %%
overexpression_causal.samples[overexpression].mean("sample").head()

# %% [markdown]
# Depending on the type of causal posterior, you can plot the outcome. The {class}`~latenta.posterior.scalar.ScalarVectorCausal` can for example plot each individual _feature_ (in this case gene) across all cells:

# %%
overexpression_causal.plot_features()

# %% [markdown]
# The plot above shows both the _median_ value of each gene across different doses of a transcription factors, together with several _credible intervals_ as shades areas. The credible interval shows, within the soft and hard priors of the model, where the actual average value of the gene expression will lie.
#
# ::::{margin}
# :::{seealso}
# https://en.wikipedia.org/wiki/Credible_interval
# :::
# ::::

# %% [markdown]
# A causal posterior can also provide a score dataframe that will rank each _feature_ (gene). In this case, this also includes columns for the maximal fold change (`fc`), absolute change (`ab`) and peak input value (`peak`).

# %%
overexpression_causal.scores

# %% [markdown]
# ### Perturbed posteriors

# %% [markdown]
# To understand whether a variable has an important impact on the downstream variable, we use perturbed posteriors. These posteriors work in a very similar way as the observed posterior, but rather than using the actual value of the variable, it uses a perturbed version. The type of perturbation varies depending on which question we want to address, and can include random shuffling, random sampling from the prior, or conditionalizing on a fixed value.
#
# While you can independently construct a perturbed posterior, you will typically create it indirectly through a causal posterior. For example, we can do random sampling from the prior using:

# %%
overexpression_causal.sample_random()

# %% [markdown]
# To check the impact of a variable on our observation, we calculate a (log-)likelihood ratio: how much more likely is the observed expression before and after we perturbed our upstream variable?

# %%
overexpression_causal.likelihood_ratio

# %% [markdown]
# These likelihood ratios were automatically added to our scores table:

# %%
overexpression_causal.scores.head()

# %% [markdown]
# :::{note}
# These likelihood ratios are mainly useful to understand the impact of a variable on an outcome *within a model*. In the model selection tutorial, we will introduce a more accurate measure to quantify whether a variable significantly affects an outcome, and whether this is significant compare to other simpler or more complex models.
# :::

# %% [markdown]
# ## Using _lacell_ to make model creation easier
#
# Specific modalities in single-cell data typically require a similar way of normalization and statistical modelling, and we have collected such prototypical models into the `lacell` package. For example, we can directly construct a model for transcriptomics from an AnnData object as follows:

# %%
import lacell as lac

transcriptome = lac.transcriptome.Transcriptome.from_adata(adata)
transcriptome.plot()

# %% [markdown]
# Note that this model is a bit more complex that the model we created before. In particular, it contains a library size normalization that will normalize the counts of each gene with a cell to the cell's total counts. However, the main ideas remain the same. Let's go through the graph from bottom to top:
#
# - We model the transcriptome as a negative binomial distributions, with a dispersion $\theta$ and mean $\mu$.
# - The mean $\mu$ is modelled as a linear combination of the relative expression in a cell, $\rho$, and its library size, $\textit{lib}$. The library size is set to the empirical library size (i.e. simply the sum of the counts in each cell).
# - The relative expression in a cell $\rho$ is itself a linear combination of the average expression of each gene in the cell, $\nu$, modelled as a latent variable, and the log-fold change $\delta$.
# - When modelling cellular processes, we typically adapt the log-fold change $\delta$. However, you can also adapt any other variables, such as the library size or dispersion, if this makes sense from a biological or technical perspective.

# %% [markdown]
# :::{seealso}
# [Why don't we just provide normalized data to latenta?](why-not-just-provide-normalized-data)
# :::

# %% [markdown]
# Once models reach a certain complexity, it becomes easier to get a variable using the {meth}`~latenta.variables.Composed.find` function, which will recursively look for an object with the given label or symbol:

# %%
foldchange = transcriptome.find("foldchange")
foldchange

# %% [markdown]
# The fold change in this case is a {class}`~latenta.modular.Additive` variable, to which you can add as many variables as you want which will all be summed:

# %%
foldchange.overexpression = la.links.scalar.Linear(
    overexpression,
    a=True,
    b=True,
    label="expression",
    definition=foldchange.value_definition,
    transforms=[la.transforms.Exp()],
)

# %%
transcriptome.plot()

# %% [markdown]
# ## More complex regression problems


# %% [markdown]
# ### Non-linear
#
# To consider non-linear relationships, we can use any of the non-linear link functions implemented in latenta:

# %% tags=["remove-input", "remove-output"]
from myst_nb import glue
logistic = la.links.scalar.Logistic._repr_formula_latex_()
glue("logistic", logistic, display=False)

spline = la.links.scalar.Spline._repr_formula_latex_()
glue("spline", logistic, display=False)

sigmoid = la.links.scalar.Sigmoid._repr_formula_latex_()
glue("sigmoid", logistic, display=False)

# %% [markdown]
# Name | Component | Formula | Description
# --- | --- | --- | ----
# Spline | {class}`~latenta.links.scalar.Spline` | {glue:math}`logistic` | A flexible non-linear function, defined by the value at a fixed set of knots
# Sigmoid | {class}`~latenta.links.scalar.Sigmoid` | {glue:math}`sigmoid` | Sigmoid kinetics model
# Switch | {class}`~latenta.links.scalar.Switch` | {glue:math}`switch` | A sudden jump

# %% [markdown]
# ### Multiple inputs
