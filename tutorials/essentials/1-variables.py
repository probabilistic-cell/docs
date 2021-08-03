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
# # Variables

# %%
import latenta as la

# %% [markdown]
# When creating a model, we define multidimensional variables, and connect these variables in a graph structure. An algorithm will then run through this graph, and will do different things depending on the type of variable we're working with, e.g. free parameters, distributions, fixed variables, ...

# %% [markdown]
# Let's use a very simple dataset as demonstration:

# %%
import scanpy as sc

adata = sc.datasets.pbmc3k()

sc.pp.filter_cells(adata, min_counts = 100)

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.filter_genes_dispersion(adata)

sc.pp.pca(adata)
sc.pp.neighbors(adata)

# %%
sc.tl.umap(adata)
sc.pl.umap(adata)

# %% [markdown]
# For demonstration purposes, we will focus only on a very broad clustering

# %%
sc.tl.leiden(adata, resolution = 0.05)
sc.pl.umap(adata, color = ["CD3D", "CD19", "CD14", "leiden"], title = ["CD3D - A T-cell marker", "CD19 - A B-cell marker", "CD14 - A monocyte marker", "leiden"])

# %% [markdown]
# ## Definition

# %% [markdown]
# A variable in latenta is a tensor, meaning it is a set of numbers that live in zero (scalar), one (vector), two (matrix) or more dimensions. Each dimension is annotated with a label, symbol and/or description, and can have further annotation.
#
# For example, a dimension can have as label "gene", symbol $g$:

# %%
genes = la.Dim(adata.var.index, id = "gene", symbol = "g")
genes

# %% [markdown]
# Similarly, for cells:

# %%
cells = la.Dim(adata.obs.index, id = "cell", symbol = "c")
cells

# %% [markdown]
# We can define how a variable should look like, without any data, using `la.Definition()`:

# %%
counts_definition = la.Definition(
    [cells, genes],
    "counts"
)
counts_definition

# %% [markdown]
# ## Fixed variables

# %% [markdown]
# Fixed variables never change.

# %%
counts = la.Fixed(adata.X, definition = counts_definition)

# %% [markdown]
# You can "run" a variable by calling it's run function

# %%
counts.run()

# %% [markdown]
# The value can be accessed as well. Because we're working with torch, this value is a `torch.tensor`:

# %%
counts.value

# %% [markdown]
# Likewise, if you want to get the value as a dataframe:

# %%
counts.value_pd

# %% [markdown]
# Do note that we can also provide pandas and xarray objects to `la.Fixed()`, and the definition of a variable will be inferred from the object's indices. This does require proper annotation of these indices, which is not the case by default in scanpy:

# %%
adata.obs.index.name = "cell"
adata.var.index.name = "gene"
cluster = la.Fixed((adata.obs["leiden"] == 0).astype(float))
cluster

# %% [markdown]
# Note that we did not provide a definition to this variable, but it was inferred from the series you provided based on the index name (`adata.obs["leiden"].index.name`). The first dimensions of both our `counts` and `leiden` are therefore equal:

# %%
cluster[0] == counts[0]

# %% [markdown]
# For discrete fixed variables, we often like to work in a "one-hot" encoding, meaning that we get a TRUE/FALSE matrix if a sample (cell) is part of a particular group (cluster). We can use `la.discrete.DiscreteFixed` to do this conversion for us by providing it with a Categorical Series:

# %% [markdown]
# ```{margin}
# A pandas categorical variable is equivalent to R's factor
# ```

# %%
adata.obs["leiden"]

# %%
leiden = la.discrete.DiscreteFixed(adata.obs["leiden"])
leiden

# %%
leiden.run()
leiden.value_pd

# %% [markdown]
# ## Parameters

# %% [markdown]
# Parameters are variables that are unknown and have to be inferred (optimized) based on our data. Parameters do require a starting default value.

# %%
lfc = la.Parameter(0., definition = la.Definition([genes]), label = "lfc", symbol = "lfc")
lfc

# %% [markdown]
# A combination of fixed and parameter variables form the leaves of our model. Any other types of variables in our models are ultimatily constructed from these leaf variables.

# %% [markdown]
# Although parameters are "free", they can still have constraints. For example, the average expression of a gene can never go below zero, as this would be non-sensical. However, most algorithms cannot directly cope with these constraints and really like to work with parameters that can go from $-\infty$ to $+\infty$ (especially if you also want flexibility). We can solve this by transforming each parameter to make sure they fit our constraints.

# %% [markdown]
# Frequently used transformations are:
#
# Description | Symbol | Transform | Description
# --- | --- | --- | ----
# All positive numbers | $R^+$ | `.Exp()` | $e^x$
# Unit interval | $[0, 1]$ | `.Logistic()` | $\frac{1}{1+e^{-x}}$
# Circular (i.e. an angle) | $[0, 2\pi[$ | `.Circular()` | atan2(y, x)

# %%
baseline = la.Parameter(
    1.,
    definition = la.Definition([genes]),
    label = "baseline",
    symbol = "baseline",
    transforms = [la.transforms.Exp()]
)
baseline

# %% [markdown]
# ## Computed

# %% [markdown]
# We use variables to compute things. In this case we create a variable that depends on other variables, which we call components:

# %%
expression = la.links.scalar.Linear(cluster, a = lfc)

# %% [markdown]
# Note that you can always plot your model:

# %%
expression.plot()

# %% [markdown]
# Components can be accessed using python's dot notation:

# %%
expression.a

# %% [markdown]
# We can sometimes also still add or change components after we have initialized the variable:

# %%
expression.b = baseline

# %%
expression.plot()

# %% [markdown]
# Frequently occuring component names are:
#
# Name | Component | Symbol | Description
# --- | --- | --- | ----
# Input | `.x` | $x$ | What you put into a computed variable
# Coefficients, weight, slope | `.a` | $a$ | How much an input affects the output
# Intercept,  bias | `.b` | $b$ | The baseline value of the output
# Skew, rate | `.skew`, `.rate` | $\gamma$ or $\lambda$ | Scales the input
# Shift | `.shift` | $\zeta$ | Shifts the input
#

# %% [markdown]
# ## Distributions

# %% [markdown]
# We often are uncertain about the value of some variables, and this uncertainty is encoded as a distribution. A distribution has two characteristics:
# - We can take a random sample. For example, we roll a dice and get 3
# - We can calculate the probability of a particular sample, also known as the likelihood. For example, in a normal dice the probability of observing a 3 is 1/6

# %% [markdown]
# A lot of commonly used distributions have an "average", also called a location, loc, mu or mean, and an "uncertainty", also called the standard deviation, sigma, scale or dispersion.

# %% [markdown]
# Distributions often have the following components:
#
# Name | Component | Symbol | Description
# --- | --- | --- | ----
# Location | `.loc` | $\mu$ | The mean, changing this parameter simply shifts the probability function
# Mean | `.mu` | $\mu$ | The mean, in case it is not a location parameter
# Scale | `.scale` | $\sigma$ | Measure of variability, changing this parameter simply scales the probability function
# Dispersion | `.dispersion` | $\theta$ | Measure of variability, in case it is not a scale parameter
# Probability | `.probs` | $p$ | A probability of success
# Total counts | `.total_counts` | $n$ | Number of tries

# %% [markdown]
# Depending on the distribution, individual elements of a sample can be dependent. This dependence can be specific to particular dimensions. For example, if we would sample from a OneHotCategorical distribution, there can only be one 1 in every row:

# %% [markdown]
# A distribution we typically use for count data is a negative binomial, and more specifically the `NegativeBinomial2` variant that has two input parameters: the average count and the dispersion.

# %% [markdown]
# ```{margin}
# The naming conventions used are those from ProbOnto (https://sites.google.com/site/probonto/home) whenever available.
# ```

# %%
counts = la.distributions.NegativeBinomial2(mu = expression)

# %%
counts.run_recursive()
counts.value_pd

# %% [markdown]
# Because we model an observation as a distribution, we can calculate how likely this observation is according to this distribution. The goal of any modelling is to maximize this likelihood, while remaining within the constraints imposed by the model.

# %%
counts.likelihood

# %% [markdown]
# ## Observations

# %% [markdown]
# Observations are fixed variables that follow a distribution. An observation always has some physical connection to reality, but the problem is that we only observe a noisy variant of this reality. This noise is not always purely technical (due to stochastic sampling of mRNAs) but also typically contains biological components (such as transcriptional bursting). Essentially, any variation that cannot be explained by the model is explained away as noise.

# %%
observation = la.Observation(
    adata.X,
    counts,
    definition = counts_definition
)

# %% [markdown]
# Running an observation does the same thing as running a fixed variable

# %%
observation.run()

# %% [markdown]
# However, we can now also calculate the (log-)likelihood of the sample

# %% 
observation.likelihood

# %% [markdown]
# ## Latent

# %% [markdown]
# Latent variables are unknown, but in contrast to a parameter they follow a distribution. Latent variables are central to probabilistic modelling, because they encompass two types of uncertainty:

# %% [markdown]
# ### Uncertainty inherent to the system
# If we take a random cell from our blood, this cell will be part of some celltype that we do not know (yet). This uncertainty is simply inherent to the population of cells. There's literally nothing we can do to reduce this uncertainty in the whole population. Instead, we model this uncertainty in the way it is, i.e. as a distribution with some known and/or unknown components.
# This type of uncertainty is often of interest, and can help us interpret the model. For example, we may be interested in how the cell type abundance change across different conditions. Other examples include:
# - The distribution of all cell's pseudotime. Are there more early than late cells? Does this change between conditions?
# - The distribution of the gene's fold changes. Are there more genes upregulated than downregulated? Are there a couple of genes with massive changes, while all other genes do not change at all?
# ````{margin}
# ```{seealso}
# https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty
# ```
# ````
# ```{note}
# In general, we call this "aleatoric uncertainty". In bayesian modelling, this type of uncertainty is typically encoded as the prior distribution.
# ````

# %% [markdown]
# ### Uncertainty because of lack of data

# If we focus on a particular cell in the blood, and we want to know its celltype. Without observing anything about this cell, our uncertainty will be equal to the uncertainty inherent to the system. However, if we would now observe some genes, our uncertainty for this cell would decrease. The more genes we observe, the more and more certain we will become.

# This type of uncertainty we want to reduce as much as possible, because it indicates that we lack data. Still, we need to model this type of uncertainty to have an idea about how certain we are and to test competing hypotheses.

# ```{note}
# In general, we call this "epistemic uncertainty". In bayesian modelling, this type of uncertainty is typically encoded as the posterior.
# ```

# %% [markdown]
# Latent variables essentially combine both types of uncertainty:

# %% [markdown]
# Name | Component | Symbol | Description
# --- | --- | --- | ----
# Prior distribution | `.p` | $p$ | The distribution followed by the variable. Note that this distribution can depend on other parameters, latent or fixed variables.
# Variational distribution | `.q` | $q$ | An approximation of the posterior distribution, i.e. the distribution followed by a variable after observing the data. The parameters of this distribution can be estimated directly (i.e. as one or more Parameter) or through the observed data through amortization

# %% [markdown]
# Sometimes components of the prior distribution may themselves depend on a latent variable. In that case, this distribution will encompass both types of uncertainty.
