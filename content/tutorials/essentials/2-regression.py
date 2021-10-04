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
# # Regression

# %%
# %load_ext autoreload
# %autoreload 2
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

# %% [markdown]
# We first define our overexpression:

# %%
overexpression = la.Fixed(adata.obs["log_overexpression"], label="overexpression")

# %% [markdown]
# We then define how this overexpression affects our gene expression on average across different cells. To keep things simple, we will first create a linear model that has both a baseline expression $b$ and a slope $a$ for each gene. Because gene expression is only defined for positive numbers, we also add an exponential transformation:

# %%
expression = la.links.scalar.Linear(
    overexpression,
    a=True,
    b=True,
    label="expression",
    definition=la.Definition([cells, genes]),
    transforms=[la.transforms.Exp()],
)

# %% [markdown]
# Although the latent slope and baseline were created automatically for us by specifying `a = True, b = True`, it's important to remember that we could have easily created these variables ourselves, for example:
#
# ```python
# slope = la.Latent(
#     p = la.distributions.Normal(scale = la.Parameter(1., transforms = [la.transforms.Exp()])),
#     definition = [genes]
# )
# ```

# %% [markdown]
# While the expression variable models our average expression, what we actually observe is a noisy sample of (UMI) counts. The prototypical way to model this is as a NegativeBinomial2. It has two components, a mean that needs to be positive (hence the exponential transformation from before) and a dispersion that also needs to be positive. This dispersion models the heterogeneity, and as this typically depends on the gene we will model it as a latent variable:

# %%
dispersion = la.Latent(
    la.distributions.LogNormal(
        la.Parameter(0.0), la.Parameter(1.0, transforms=[la.transforms.Exp()])
    ),
    definition=la.Definition([genes]),
)

# %% [markdown]
# Note the LogNormal prior here: this distribution has a _support_ only positive numbers, and latenta will automatically try to match this support in the variational distribution $q$, in this case by adding an exponention transform:

# %%
dispersion.q

# %% [markdown]
# We can now define the distribution that our data will follow:

# %%
transcriptome_p = la.distributions.NegativeBinomial2(
    mu=expression, dispersion=dispersion
)

# %% [markdown]
# And then we can define our observation:

# %%
transcriptome = la.Observation(
    adata.X,
    transcriptome_p,
    definition=[cells, genes],
    label="transcriptome",
)

# %%
transcriptome.plot()

# %% [markdown]
# Note the many free parameters that form the leaves of our model. These will have to be estimated by the model. But first, why are there so many parameters even for the such a simple linear regression?
#
# Let's remind ourselves what we're actually trying to accomplish. We're trying to create a good model of our observations. There are however many models that will provide a very good fit of the data equally. For example, we could just give the actual count matrix as input to the negative binomial and this trivial model would fit extremely well.
#
# So we don't just want a model. We want a model that can explain our observation well, while being both generalizeable and interpretable. And to accomplish this, we have to limit the flexibility that our model can have. You have already done this in two ways:
#
# - Hard priors are those that completely constrain the model. For example, by specifying a linear function we don't allow any non-linearities. There's simply no way that the model moves beyond these constraints.
# - Soft priors are those that only push the latent variables towards more likely values. For example, we want to discourage extreme slopes that are far away from 0, unless the data really provides strong evidence for these extreme slopes
#
# The purpose of these parameters is to balance the wishes of the soft priors to the wishes of the observations. The parameters of the variational distributions $q$ will try to explain the observations while also remaining faithful to the prior distributions $p$. The parameters of $p$ on the other hand will try to accomodate the parameters of $q$ as well as possible, but it cannot do this perfectly as these parameters are shared across all genes. It's this pushing and pulling between priors and variational distributions that prevent overfitting and underfitting of the model. At the same time, through $q$, we get some estimates of the uncertainty of our latent variables for free!

# %% [markdown]
# Mathematically speaking, the "wishes of the observations" is called the **likelihood** and noted by $P(x|z)$, where $x$ are the observations and $z$ the latent variables. The "wishes of the prior" on the other hand is called the **prior probability** and noted by $P(z)$.

# %% [markdown]
# To infer an optimal value for these parameters, we have to find a solution that best balances the needs of the prior distribution with those of the observations. And one of the fastest ways to do that is to use gradient descent, which starts from an initial value and then tries to move these initial values slowly but surely into better values. These tasks are fullfilled by a loss function (`ELBO`), an optimizer (`Adam`), and an overarching training class (`SVI`):

# %%
inference = la.infer.svi.SVI(
    transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)

# %% [markdown]
# We can now train for a couple of iterations or until the loss function has reached convergence:

# %%
trace = trainer.train(10000)
trace.plot()

# %% [markdown]
# You can check that our values have changed:

# %%
transcriptome.p.mu.a.q.loc.run_local()
transcriptome.p.mu.a.q.loc.value_pd.head()

# %% [markdown]
# Our inference algorithm did not fullfill all 10000 iterations, but has permaturely stopped as it has detected _convergence_. Do note that this converge is not always perfect, and we will see later that there are some circumstances where further training may be advisable.

# %% [markdown]
# ## Interpreting a regression model
#
# For interpretation of the model, we can then use 3 main types of posteriors:

# %% [markdown]
# ### Observed posteriors
#
# Because we are working with a probabilistic model, every time we run through the model our results will change. For example, each time we sample from variational distribution of the slope $q$ the output will be different, which will affect any downstream variables even if they are themselves deterministic. To interpret the results, we thus have to sample multiple times from the model, using a {class}`~la.posterior.Observed` posterior:

# %%
transcriptome_observed = la.posterior.Observed(
    transcriptome, retain_samples={expression.a}
)
transcriptome_observed.sample(5)

# %% [markdown]
# `.samples` is a dictionary containing the samples of each variable that was upstream of the transcriptome (if they were provided in the `retain_samples` list). We can access each variable either by providing the variable itself:

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

# %% [markdown]
# For example, we can get and rank by mean slope as follows:

# %%
mean_slopes = (
    transcriptome_observed.samples[expression.a]
    .mean("sample")
    .to_pandas()
    .sort_values()
)

scores = mean_slopes.rename("slope").to_frame()
scores["symbol"] = adata.var["symbol"][scores.index]
scores

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
# This posterior also contains samples, but now for some _pseudocells_ with each a distinct value for the `overexpression` variable:

# %%
overexpression_causal.samples[overexpression].mean("sample")

# %% [markdown]
# Depending on the type of causal posterior, you can plot the outcome. The {class}`~latenta.posterior.scalar.ScalarVectorCausal` can for example plot each individual _feature_ across all cells (in this case gene):

# %%
overexpression_causal.plot_features()

# %% [markdown]
# This plot shows both the _median_ value of each gene across different doses of a transcription factors, together with several _credible intervals_ as shades areas. The credible interval shows, within the constraints of soft and hard priors, where the actual average value of the gene expression will lie.
#
# ::::{margin}
# :::{seealso}
# https://en.wikipedia.org/wiki/Credible_interval
# :::
# ::::

# %% tags=["remove-input", "remove-output"]
from myst_nb import glue

glue(
    "conditional",
    la.utils.latex.convert_mathtex_to_latex(
        f"$P({transcriptome.p.mu.symbol}|{overexpression.symbol} = ...)$"
    ),
)

# %% [markdown]
# Mathematically, this plot represents the conditional posterior:
#
# $$P(\mu|\text{overexpression} = ...)$$

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
# Mathematically, these values represents the ratio between posteriors probabilities:
#
# $$\frac{P(\mu|\text{overexpression})}{P(\mu|\text{overexpression}_\text{shuffled})}$$

# %% [markdown]
# These likelihood ratios were also automatically added to our scores table:

# %%
overexpression_causal.scores.head()

# %% [markdown]
# :::{note}
# These likelihood ratios are mainly useful to understand the impact of a variable on an outcome *within a model*. In the model selection tutorial, we will introduce a better measure to quantify whether a variable **significantly** affects an outcome, by comparing it to other simpler or more complex models.
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
# Note that this model is a bit more complex that the model we created before. In particular, it contains a library size normalization that will normalize the counts of each gene with a cell to the cell's total counts. However, the main ideas remain the same:
#
# - We model the counts as a negative binomial distributions, with a dispersion $\theta$ and mean $\mu$.
# - The mean $\mu$ is modelled as a linear combination of the relative expression in a cell $\rho$, and its library size $\mathit{lib}$. The library size is set to the empirical library size (i.e. simply the sum of the counts in each cell).
# - The $\rho$ itself is a linear combination of the average expression of each gene in the cell $\nu$, modelled as a latent variable, and the log-fold change $\delta$.
# - The log-fold change $\delta$ is a modular variable.
# - When modelling cellular processes, we typically add things to the log-fold change $\delta$. However, you can also adapt any other variables, such as the library size or dispersion, if this makes sense from a biological or technical perspective.

# %% [markdown]
# :::{seealso}
# [Why don't we just provide normalized data to latenta?](why-not-just-provide-normalized-data)
# :::

# %% [markdown]
# Once models reach a certain complexity, it becomes easier to get a variable using the {meth}`~latenta.variables.Composed.find` function, which will recursively look for an object with the given label or symbol:

# %%
foldchange = transcriptome.find("foldchange")

# %% [markdown]
# Let's add the overexpression to the fold change:

# %%
foldchange.overexpression = la.links.scalar.Linear(
    overexpression, a=True, label="expression", definition=foldchange.value_definition
)

# %% [markdown]
# Note that we do not transform this variable, nor do we add a baseline (or intercept) to this function as this is all handled downstream by the $\rho$ function:

# %%
transcriptome.plot()

# %% [markdown]
# ## More complex regression problems


# %% [markdown]
# ### Non-linear
#
# To consider non-linear relationships, we can use any of the non-linear link functions implemented in latenta:

# %% tags=["remove-input"]
import pandas as pd
import IPython.display

links = [la.links.scalar.Logistic, la.links.scalar.Spline, la.links.scalar.Sigmoid]

link_table = []
for link in links:
    link_table.append(
        {
            "name": link.__name__,
            "reference": "`" + link.__module__ + "." + link.__name__ + "`",
            "description": link.__doc__.strip().split("\n")[0],
        }
    )
link_table = pd.DataFrame(link_table)
IPython.display.HTML(link_table.to_html())

# %% [markdown]
# We'll illustrate {class}`latenta.links.scalar.spline.Spline`.

# %%
foldchange = transcriptome.find("foldchange")

# %%
foldchange.overexpression = la.links.scalar.Spline(
    overexpression,
    label="expression",
    definition=foldchange.value_definition,
    transforms=[la.transforms.Exp()],
)

# %%
foldchange.plot()

# %%
inference = la.infer.svi.SVI(
    transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)

# %%
trace = trainer.train(10000)
trace.plot()

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression, transcriptome
)
overexpression_causal.observed.sample()
overexpression_causal.sample(30)
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features()

# %% [markdown]
# ### Multiple inputs
