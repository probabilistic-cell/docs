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
# # Variables

# %%
import latenta as la

# %% [markdown]
# A model is a collection of variables together with a description of how these variables are related. Here, we will first look at the basic types of variables in a model, and how we can connect them with a graph structure.

# We'll use a small single-cell transcriptomics dataset as demonstration:

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
# Just for demonstration, we'll focus on a broad clustering

# %%
sc.tl.leiden(adata, resolution = 0.05)
sc.pl.umap(adata, color = ["CD3D", "CD19", "CD14", "leiden"], title = ["CD3D - A T-cell marker", "CD19 - A B-cell marker", "CD14 - A monocyte marker", "leiden"])

# %% [markdown]
# ## Definition

# %% [markdown]
# A variable in latenta is mathemtiacally represented by a tensor, meaning it is a set of numbers that live in zero (scalar), one (vector), two (matrix) or more dimensions. Each dimension of a variable has coordinates (`.coords`), and identifier (`.id`). Furthermore,a dimension can also be annotation with a label, symbol and/or description.
#
# A "genes" dimension could be defined as follows:

# %%
genes = la.Dim(adata.var.index, id = "gene")
genes

# %% [markdown]
# We can also simply provide a pandas series/index, although we have to take care to set the `index.name`:

# %%
adata.var.index.name = "gene"
genes = la.Dim(adata.var.index)
genes

# %% [markdown]
# Similarly, for cells:

# %%
adata.obs.index.name = "cell"
cells = la.Dim(adata.obs.index, id = "cell")
cells

# %% [markdown]
# We can define how a variable should look like, without any data, using {class}`la.Definition()`:

# %%
counts_definition = la.Definition(
    [cells, genes],
    "counts"
)
counts_definition

# %% [markdown]
# ## Fixed variables

# %% [markdown]
# Fixed variables contain a tensor that never changes:

# %%
counts = la.Fixed(adata.X, definition = counts_definition)

# %% [markdown]
# You can _run_ a variable by calling it's run function:

# %%
counts.run()

# %% [markdown]
# The value can be accessed as well. Because we're working with torch, this value is a `torch.tensor`:

# %%
counts.value

# %% [markdown]
# Because our variable is annotated, we can also get this same value as a dataframe:

# %%
counts.value_pd

# %% [markdown]
# Do note that we can also provide pandas and xarray objects to :class:`Fixed`, and the definition of a variable will be inferred from the object's indices (if we gave them proper names).

# %%
cluster = la.Fixed((adata.obs["leiden"] == 0).astype(float), label = "cluster")
cluster

# %% [markdown]
# Note that we did not provide a definition to this variable, but it was inferred from the series you provided based on the index name (`adata.obs["leiden"].index.name`). The first dimensions of both our `counts` and `leiden` are therefore equal:

# %%
cluster[0] == counts[0]

# %% [markdown]
# For discrete fixed variables, we often like to work in a "one-hot" encoding, meaning that we get a binary (True/False) matrix if a sample (cell) is part of a particular group (cluster). We can use `la.discrete.DiscreteFixed` to do this conversion by providing it with a categorical pandas series:

# %% [markdown]
# ```{margin}
# A pandas categorical series is equivalent to R's factor
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
# Although parameters are _free_, they can still have constraints. For example, the average expression of a gene can never go below zero, as this would be non-sensical. However, most algorithms cannot directly cope with these constraints and really like to work with parameters that can go from $-\infty$ to $+\infty$ (especially if you also want flexibility). We can solve this by transforming each parameter to make sure they fit our constraints.

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
expression = la.links.scalar.Linear(cluster, a = lfc, label = "expression")

# %%
expression

# %% [markdown]
# Both dimensions of this variable are **broadcasted**, meaning that their coordinates and size are set by upstream components.

# %%
expression["cell"]

# %% [markdown]
# A critical tool when building complex models is to plot how different variables are related in the graph structure:

# %%
expression.plot()

# %% [markdown]
# :::{seealso}
# Visualization and inspection of models is further discussed in the [guide](/guide/inspection)
# :::

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
# We are often uncertain about the value of some variables, and this uncertainty is encoded as a distribution. A distribution has two characteristics:
# - We can take a random sample. For example, we roll a dice and get 3
# - We can calculate the probability of a particular sample, also known as the likelihood. For example, in a normal dice the probability of observing a 3 is 1/6

# %% [markdown]
# A lot of commonly used distributions have an _average_, also called a location, loc, mu or mean, and an _uncertainty_, also called the standard deviation, sigma, scale or dispersion.

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

observation.plot()

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
# Latent variables are unknown. but in contrast to a parameters they follow a distribution. Latent variables are central to probabilistic modelling, because they encompass two types of uncertainty:

# %% [markdown]
# ### Uncertainty inherent to the system
# ::::{margin}
# :::{note}
# In general, we call this _aleatoric uncertainty_. In bayesian modelling, this type of uncertainty is typically encoded as the prior distribution.
# :::
# ::::
# 
# Let's say we randomly take a cell from a tissue. Until we have observed something of this cell, we don't know anything what type of cell it is, except perhaps that some cell types are more likely because they are more abundant. This uncertainty is simply inherent to the population, and nothing we do can change that. We model this uncertainty in the way it is: as a probability distribution with some known or unknown upstream components.
# 
# This type of uncertainty is often of interest, and can provide  some interesting biological information in more complex models. For example:
# - How does cell type abundance change across different conditions?
# - The distribution of all cell's pseudotime. Are there more early than late cells? Does this change between conditions?
# - The distribution of the gene's fold changes. Are there more genes upregulated than downregulated? Are there a couple of genes with massive changes, while all other genes do not change at all?

# %% [markdown]
# ### Uncertainty because of lack of data
#
# ::::{margin}
# :::{note}
# In general, we call this _epistemic uncertainty_. In bayesian modelling, this type of uncertainty is typically encoded as the posterior.
# :::
# ::::
# Let's say we focus on one particular cell, and we want to know it's celltype. Naturally, if we don't observe anything about this cell, our uncertainty will be the same as that inherent to the system as described above. However, if we would now observe some gene expression, our uncertainty for this particular cell will decrease. The more genes we observe, the more certain we will become.
#
# This type of uncertainty is not inherent to the cell. The cell does not change its celltype just because we are uncertain about it. It therefore does not have a direct connection to reality, but is simply an artefact of us knowing not enough. Nonetheless, modelling this uncertainty is crucial because it gives us an idea about how certain we are about a variable. For example:
# - We may be very certain about a cell's cell type, but are very uncertain about the cellular state. This could tell us that we don't have enough sequencing depth to assign a cell's state.
# - If a gene's fold-change can be both negative, positive and close to 0, we know we don't have enough data. This doesn't mean the gene is not differentially expressed, it simply means we don't have enough data know whether it is.
# 
# :::{seealso}
# https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty
# :::

# %% [markdown]
# Latent variables combine both types of uncertainty:

# %% [markdown]
# Name | Component | Symbol | Description
# --- | --- | --- | ----
# Prior distribution | `.p` | $p$ | The distribution followed by the variable. Note that this distribution can depend on other parameters, latent or fixed variables.
# Variational distribution | `.q` | $q$ | An approximation of the posterior distribution, i.e. the distribution followed by a variable after observing the data. The parameters of this distribution can be estimated directly (i.e. as one or more Parameter) or through the observed data through amortization

# A prototypical example of a latent variable is the slope (i.e. fold-change) in a linear model:

# %%
lfc_p = la.distributions.Normal(
    la.Parameter(0.),
    la.Parameter(1., transforms = [la.transforms.Exp()])
)
lfc = la.Latent(
    lfc_p,
    definition = la.Definition([genes])
)
expression = la.links.scalar.Linear(cluster, a = lfc, label = "expression")
expression.plot()

# %% [markdown]
# Note that many link function allow you to automatically create a latent variable, although you have to provide the correct definition if some dimensions cannot be inferred from other components:

# %%
expression = la.links.scalar.Linear(cluster, a = True, label = "expression", definition = la.Definition([cells, genes]))
expression.plot()

# %%


# %%
expression = la.links.scalar.Linear(cluster, a = lfc, label = "expression")

# %% [markdown]
# Sometimes components of the prior distribution may themselves depend on a latent variable. In that case, this distribution will encompass both types of uncertainty.
