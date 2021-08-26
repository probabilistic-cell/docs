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
import scanpy as sc


# %% [markdown]
# Modelling typically consists of several steps:
# 1. Creating known and unknown variables
# 2. Connect these variables in a graph structure
# 3. Infer the value of the unknown variables
# 4. Interpret the model
#
# In this tutorial, we give a brief overview of the first two steps. More details on using latenta to interpret specific types of datasets can be found in the [tutorials](/tutorials), while detailed explanations on specific problems (e.g. the cell cycle) can be found in the [user guide](/guide).
#
# While we will use a simple single-cell transcriptomics dataset as demonstration, in these tutorials we will primarily focus on the core that are relevant to any type of modality, latent space and experimental design.

# %%
adata = la.data.load_myod1()

# %% tags=["hide-input", "remove-output"]
import numpy as np

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
# In this data, we have overexpressed the transcription factor *Myod1* or a control (mCherry) in a stem cell line across different batches. 
#
# As output, we get:
# * the transcriptome 
# * what was overexpressed (*gene overexpressed*)
# * the level of overexpression (*log_overexpression*)

# %% [markdown]
# ## General definition of a variable

# %% [markdown]
# A variable in latenta is a representation of a tensor, meaning it is a set of numbers that live in zero (scalar), one (vector), two (matrix) or more dimensions. Each dimension of a variable has coordinates (`.coords`), and identifier (`.id`). Furthermore, a dimension can also be annotation with a label, symbol and/or description.
#
# Let's start with the prototypical variable, a count matrix, which contains the information of the genes expressions in all the cells. 
#
# To define its structure we will first need to define its two dimensions: *genes* and *cells*. 
# The *genes*, for example, can be defined as follows:

# %%
genes = la.Dim(adata.var.index, id="gene")
genes

# %% [markdown]
# We can also simply provide a pandas series/index, although we have to take care to set the `index.name`:

# %%
adata.var.index.name = "gene"
genes = la.Dim(adata.var.index)
genes

# %% [markdown]
# Similarly, for *cells*:

# %%
adata.obs.index.name = "cell"
cells = la.Dim(adata.obs.index, id="cell")
cells

# %% [markdown]
# Now that we have the two dimensions `genes` and `cells`, we can define how the structure of a count matrix should look like using {class}`la.Definition()`:

# %%
counts_definition = la.Definition([cells, genes], "counts")
counts_definition

# %% [markdown]
# ## 1. Fixed variables

# %% [markdown]
# Fixed variables contain a tensor with values that never changes.  
#
# Count matrix is a good example of a fixed variable. Let's initialise a variable `counts` containing the transcriptome from our data using `la.Fixed()` and by specifying its structure with `counts_definition`: 

# %%
counts = la.Fixed(adata.X, definition=counts_definition)

# %% [markdown]
# You can _run_ a variable by calling it's run function:

# %%
counts.run()

# %% [markdown]
# The value can also be accessed and because we're working with torch, this value is a `torch.tensor`:

# %%
counts.value

# %% [markdown]
# Because our variable is annotated, we can also get this same value as a dataframe:

# %%
counts.value_pd.head()

# %% [markdown]
# Do note that we can also provide pandas and xarray objects to {class}`Fixed`, and the definition of a variable (its dimensional structure) will be inferred from the object's indices (if we gave them proper names).  
#
# Let's try with the level of overexpression in our data:

# %%
adata.obs["log_overexpression"]

# %%
overexpression = la.Fixed(
    adata.obs["log_overexpression"], label="overexpression", symbol="overexpression"
)
overexpression

# %% [markdown]
# As we can see we didn't provide a definition to this variable, but it was inferred from the series you provided based on the index name (`adata.obs["log_overexpression"].index.name`). The first dimensions of both our `counts` [cells, genes] and `overexpression` [cells] are therefore equal:

# %%
counts[0] == overexpression[0]

# %% [markdown]
# For discrete fixed variables, we often like to work in a "one-hot" encoding, meaning that we get a binary (True/False or 1/0) matrix if a sample (cell) is part of a particular group (cluster). We can use `la.variables.DiscreteFixed` to do this conversion by providing it with a categorical pandas series. In our data, an example of discrete fixed variable would be a variable indicating what (*Myod1* or control mCherry) is overexpressed in each cell:

# %% [markdown]
# ```{margin}
# A pandas categorical series is equivalent to R's factor
# ```

# %% tags=[]
adata.obs["gene_overexpressed"]

# %% [markdown]
# Lets now create a discrete fixed variable called `overexpressed` using `la.variables.DiscreteFixed``

# %%
overexpressed = la.variables.DiscreteFixed(adata.obs["gene_overexpressed"])
overexpressed


# %%
overexpressed.run()
overexpressed.value_pd.head()

# %% [markdown]
# ## 2. Parameters

# %% [markdown]
# Parameters are variables that are unknown and have to be inferred (optimized) based on our data and they require a starting default value. An example is the log fold-change of a gene between two categories which we can initialise at 0.

# %%
lfc = la.Parameter(0.0, definition=la.Definition([genes]), label="lfc", symbol="lfc")
lfc

# %% [markdown]
# A combination of fixed and parameter variables form the leaves of our model. Any other types of variables in our models are ultimatily constructed from these leaf variables.

# %% [markdown]
# Although parameters are _free_, not fixed, they can still have constraints. For example, the average expression of a gene can never go below zero, as this would be non-sensical. However, most algorithms cannot directly cope with these constraints and really like to work with parameters that can go from $-\infty$ to $+\infty$ (especially if you also want flexibility). We can solve this by transforming each parameter to make sure they fit our constraints.

# %% [markdown]
# Frequently used transformations are:
#
# Description | Support | Transform | Formula
# --- | --- | --- | ----
# All positive numbers | $R^+$ | `.Exp()` | $e^x$
# Unit interval | $[0, 1]$ | `.Logistic()` | $\frac{1}{1+e^{-x}}$
# Circular (i.e. an angle) | $[0, 2\pi[$ | `.Circular()` | $atan2(y, x)$
# Simplex | $\in {0, 1} \wedge \sum = 1$ | `.Softmax()` | $\frac{e^{x_i}}{\sum_i e^{x_i}}$

# %% [markdown]
# More concretely we can define a parameter `baseline` which is the baseline expression of the genes and as just mentioned it would not make sense to have a negative baseline expression we apply an exponential transformation using `la.transforms.Exp` that we specifiy to the argument transforms of `la.Parameter`:

# %%
baseline = la.Parameter(
    1.0,
    definition=la.Definition([genes]),
    label="baseline",
    symbol="baseline",
    transforms=[la.transforms.Exp()],
)
baseline

# %% [markdown]
# ## 3. Computed variables

# %% [markdown]
# The different variables are also used to compute "new" information about our data or to generalise certain results to any future cells as explained in 0-modeling. In this case we create a variable that will depend on other variables, called components. Most of the time this is done thanks to a regression using a/using linear model and link function applied to modify the input data (see part 2. of the tutorial).
#
# For example, we can model the expression $y$ of a gene $g$ in a cell $c$ as as factor (`lfc`) times how much *Myod1* is overexpressed (`overexpression`) in that cell $y_{g,c}$ ~ `lfc`$_{g}$*`overexpression`$_{c}$ using `la.links.scalar.Linear`:

# %%
import matplotlib.pyplot as plt
plt.plot(np.linspace(0, 2, 9), np.linspace(0, 2, 9))
plt.show() 

# %%
expression = la.links.scalar.Linear(overexpression, a=lfc, label="expression")
expression

# %% [markdown]
# Both dimensions of this variable are **broadcasted**, meaning that their coordinates and size are set by upstream components (signified by a •).
#
# The actual definition of the variable can be obtained using `.value_definition`:

# %%
expression.value_definition

# %% [markdown]
# A critical tool when building complex models is to plot how different variables are related in the graph structure:

# %%
expression.plot()

# %% [markdown]
# :::{seealso}
# Visualization and introspection of models is further discussed in the [guide](/guide/introspect)
# :::

# %% [markdown]
# Components can be accessed using python’s dot notation:

# %%
expression.a

# %% [markdown]
# Often, you can still add or change components after we have initialized the variable for example we want to add an intercept or baseline such that $y_{g,c}$ ~ `lfc`$_{g}$*`overexpression`$_{c}$ + $baseline_{g}$

# %%
expression.a

# %%
expression.b = baseline

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
# ## 4. Distributions

# %% [markdown]
# We are often uncertain about the value of some variables, and this uncertainty is encoded as a distribution. 
# There is two ways to define/initialize a distribution:
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
# Depending on the distribution, individual elements of a sample can be dependent. This dependence can be specific to particular dimensions. For example, if we would sample from a OneHotCategorical distribution, there can only be one 1 in every row.

# %% [markdown]
# For example we previously defined the computed variable `expression`, however we cannot be certain that the expression will be this exact value, but it will rather range around this value. A distribution we typically use for gene expression is a negative binomial, and more specifically the `NegativeBinomial2` variant that has two input parameters: the average count (`.mu`) and the dispersion (`.dispersion`). (Note that we denote the fact that it is a distribution variable with _p)

# %% [markdown]
# ```{margin}
# The naming conventions used are those from ProbOnto (https://sites.google.com/site/probonto/home) whenever available.
# ```

# %%
transcriptome_p = la.distributions.NegativeBinomial2(mu=expression)

# %% [markdown]
# Running a distribution will take a sample from it/pick a certain value from the distribution:

# %%
transcriptome_p.run()
transcriptome_p.value_pd.head()

# %% [markdown]
# Because we model an observation as a distribution, we can calculate how likely this observation is according to this distribution. The goal of any modelling is to maximize this likelihood, while remaining within the constraints imposed by the model.

# %%
transcriptome_p.likelihood

# %% [markdown]
# ## 5. Observations

# %% [markdown]
# Observations are fixed variables that follow a distribution. An observation always has some physical connection to reality, but the problem is that we only observe a noisy variant of this reality. This noise is not always purely technical (due to stochastic sampling of mRNAs) but also typically contains biological components (such as transcriptional bursting). Essentially, any variation that cannot be explained by the model is explained away as noise. In our case, this noise is contained in the dispersion component of the Negative Binomial.

# %%
transcriptome = la.Observation(value=counts.value, p=transcriptome_p, definition=counts_definition)

transcriptome.plot()

# %% [markdown]
# Running an observation does the same thing as running a fixed variable

# %%
transcriptome.run()

# %% [markdown]
# Because we model an observation as a distribution, we can calculate how likely this observation is according to this distribution. The goal of modelling is to maximize this likelihood, while remaining within the constraints imposed by the model.

# %%
transcriptome.likelihood

# %% [markdown]
# ## 6. Latent

# %% [markdown]
# Latent variables are unknown as parameter but they follow distribution rather than having one fixed value. Latent variables are central to probabilistic modelling, because they encompass two types of uncertainty:
#
# ::::{margin}
# :::{seealso}
# https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty
# :::
# ::::

# %% [markdown]
# ### 1. Uncertainty inherent to the system
# ::::{margin}
# :::{note}
# In general, we call this _aleatoric uncertainty_. In bayesian modelling, this type of uncertainty is typically encoded as the prior distribution.
# :::
# ::::
#
# Let's say we are randomly sampling a cell from a tissue. Every time we take such a sample, we don't have an idea about what type of cell it will be, except perhaps that some cell types are more likely to be picked because they are more abundant. This uncertainty is inherent to the population, and nothing we do can change that. We model this uncertainty as an appropriate probability distribution, which can have some known or unknown components.
#
# This type of uncertainty is often of interest, and can provide some interesting biological information in more complex models. 
#
# For example:
# - How does cell type abundance change across different conditions?
# - The distribution of all cell's pseudotime. Are there more early than late cells? Does this change between conditions?
# - The distribution of the gene's fold changes. Are there more genes upregulated than downregulated? Are there a couple of genes with massive changes, while all other genes do not change at all?
# - The effect of a transcription factor. Does it have many target genes with a subtle effect? Or a few target genes but with a very strong effect?

# %% [markdown]
# ### 2. Uncertainty because of lack of data
#
# ::::{margin}
# :::{note}
# In general, we call this _epistemic uncertainty_. In bayesian modelling, this type of uncertainty is typically encoded as the posterior.
# :::
# ::::
# Let's say we focus on one particular cell, and we want to know it's cell type. Naturally, before we have observed anything about this cell, our uncertainty will be the same as that inherent to the system (as described above). However, if we would now observe some gene expression, our uncertainty for this particular cell will decrease. The more genes we observe, the more certain we will become.
#
# This type of uncertainty is not inherent to the cell. The cell does not change its cell type just because we are uncertain about it. Therefore it does not have a direct connection to reality, but is simply an artefact of us not knowing enough. Nonetheless, modelling this uncertainty is crucial because it gives us an idea about how certain we are about a variable. For example:
# - We may be very certain about a cell's cell type, but be very uncertain about the cellular state. This could tell us that we don't have enough sequencing depth to assign a cell's state.
# - If a gene's fold-change has distribution spreading over both negative and positive we might not have enough data. This doesn't mean necessarily mean that the gene is not differentially expressed, but rather that we don't have enough data know whether it is.
# - It may be that two transcription factors can equally regulate gene expression??? in the model, and that we don't have enough data to say which one is more likely.

# %% [markdown]
# ### Constructing a latent variable
#
# Latent variables combine both types of uncertainty:

# %% [markdown]
# Name | Component | Symbol | Description
# --- | --- | --- | ----
# Prior distribution | `.p` | $p$ | The distribution followed by the variable, inherent to the system. Note that this distribution can depend on other parameters, latent or fixed variables.
# Variational distribution | `.q` | $q$ | An approximation of the posterior distribution, i.e. the distribution followed by a variable after observing the data. The parameters of this distribution can be estimated directly (i.e. as one or more Parameter) or through the observed data using amortization
#
# A prototypical example of a latent variable is the slope (i.e. fold-change) in a linear model. 

# %%
lfc_p = la.distributions.Normal(
    loc=0.0, scale=la.Parameter(1.0, transforms=[la.transforms.Exp()])
)
latent_lfc = la.Latent(p=lfc_p, definition=la.Definition([genes]), label="lfc")
latent_lfc.plot()

# %% [markdown]
# The prior distribution $p$ in this case is a normal distribution with one free parameter: the scale $\sigma$. This parameter determines how far the slope _on average_ can be different than 0. The only reason we can estimate this as a parameter is because we are pooling information across many genes. This kind of {term}`multi-level modelling` is very powerful, as it includes multiple testing correction directly within the model {citel}`gelman_why_2009`.
#
# The variational distribution $q$ on the other hand contains two parameters both specific for each gene. These contain our idea of where the slope of a gene will lie: the location $\mu$ is the average, while the scale $\sigma$ our uncertainty.

# %% [markdown]
# We can now modify the lfc or slope (`.a`) of the computed variable `expression` with a latent variable rather than a parameter.

# %%
expression.a=latent_lfc
expression.plot()

# %% [markdown]
# Note that many link functions will create a latent variable automatically if you specify `True`, although you have to provide the correct definition if some dimensions cannot be inferred from other components. For example for the expression that we defined earlier on, 

# %%
expression = la.links.scalar.Linear(
    overexpression,
    a=True,  # 👈
    b=baseline,
    label="expression",
    definition=la.Definition([cells, genes]),  # ⚠️
)
expression.plot()

# %% [markdown]
# :::{note}
# Sometimes components of the prior distribution may themselves depend on a latent variable. In that case, this distribution will encompass both types of uncertainty. This would be the case for [most examples we gave for prior distributions](#uncertainty-inherent-to-the-system).
# :::

# %% [markdown]
# Just like parameters, the two distributions of latent variables may also have transformations. This is for example the case if we would use a `.LogNormal` as prior, which only has a support on positive numbers. A good example is the baseline expression of the genes which cannot take negative values:

# %%
baseline_p = la.distributions.LogNormal(
    loc=la.Parameter(0.0), scale=la.Parameter(1.0, transforms=[la.transforms.Exp()])
)
latent_baseline = la.Latent(p=baseline_p, definition=la.Definition([genes]), label="baseline")
latent_baseline.plot()

# %% [markdown]
# We can now modify the baseline (`.b`) of expression as a latent variable:

# %%
expression.b=latent_baseline
expression.plot()

# %% [markdown]
# Should we specify that it doesn't automatically updates the other variables that depends on them?

# %%
transcriptome.plot()

# %% [markdown]
# ## Main points
# - Modelling is creating and connecting variables in a graph structure
# - Leaf variables are the start of any model. Different types of leaf variables have different functions:
#    - Fixed variables are provided by you or set to a reasonable default value
#    - Parameters are free and have to be estimated from the data
#    - Observations are fixed variables that follow a distribution
# - Computed variables link different variables together in a deterministic way
# - Distributions model uncertainty
# - Latent variables are the variables we're most interested in, as they encompass both uncertainty inherent to the system and uncertainty due to the lack of data
#
# Once we have specified a model, the next question is how we can find optimal values for the free parameters.

# %%
