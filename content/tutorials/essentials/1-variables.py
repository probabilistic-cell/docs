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
# %load_ext autoreload
# %autoreload 2
import latenta as la
import scanpy as sc


# %% [markdown]
# In the essentials tutorials, we will work with a small but nonetheless interesting single-cell transcriptomics dataset:

# %%
adata = la.data.load_myod1()

# %%
adata.obs

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
# In this data, the transcription factor *Myod1* or a control (mCherry) have been overexpressed in a stem cell line across different batches.
#
# As output, we get:
# * the transcriptome
# * what is overexpressed (*gene overexpressed*)
# * the level of overexpression (*log_overexpression*)
# * the batch

# %% [markdown]
# With this data set we can think of many interesting questions, here are three rather simple ones that we will adress in this tutorial:
# * Does the overexpression of *Myod1* impact the expression of other genes? Of which genes? What is the relationship between the overexpression and the expression of the impacted genes?
# * Can we use this data to learn more about myogenesis?
# * Are some cells cycling? Is there any interaction with the differentiation process? 
#
# Along this tutorial we will learn how to build more or less complex model to answer these questions, we will also learn how to correct for technical biais such as the batch, how to get  some interpretation about the models, and compare them. 

# %% [markdown]
# ## General definition of a variable

# %% [markdown]
# Variables represent quantities/information that vary in the system we are interested in. They can be known (in our case, for example, we know the transcriptome, the overexpression, etc.) or need to be inferred from the data (the fold change of expression of a gene, the position of a cell along myogenesis, etc.). Variables represent the different units of the models we will construct. Indeed a statistical model is a mathematical relationship between variables. We emit an hypothesis on how the variables might interact and we use the data to validate it and/or to learn about some unknown variables.

# %% [markdown]
# In latenta, variables are represented as a tensor, meaning it is a set of numbers that live in zero (scalar), one (vector), two (matrix), or more dimensions that contain some type of information about our data. Each dimension of a variable has coordinates (`.coords`) and an identifier (`.id`). Furthermore, a dimension can also be annotated with a label, symbol, and/or description.
#
# Let's start with the prototypical variable in single-cell RNA-seq, a count matrix, which contains the raw genes expressions in all the cells.
#
# To define its structure, we first need to define its two dimensions: *genes* and *cells*.
# The *genes*, for example, can be defined as follow:

# %%
genes = la.Dim(adata.var.index, name="gene")
genes

# %% [markdown]
# We can also only provide a pandas series/index, although we have to take care to set the `index.name` beforehand:

# %%
adata.var.index.name = "gene"
genes = la.Dim(adata.var.index)
genes

# %% [markdown]
# Similarly, for *cells*:

# %%
adata.obs.index.name = "cell"
cells = la.Dim(adata.obs.index, name="cell")
cells

# %% [markdown]
# Now that we have the two dimensions `genes` and `cells`, we can define how the structure of a count matrix should look like using {class}`la.Definition()`:

# %%
counts_definition = la.Definition([cells, genes], "counts")
counts_definition

# %% [markdown]
# Let's now get more into the different types of variables that exist in latenta.

# %% [markdown]
# ## 1. Fixed variables

# %% [markdown]
# Fixed variables contain a tensor with values that never change.
#
# The count matrix is an excellent example of a fixed variable. Let's initialise the variable `counts` containing the raw counts of our data using `la.Fixed()` and by specifying its structure with `counts_definition`:

# %%
counts = la.Fixed(adata.X, definition=counts_definition)

# %% [markdown]
# You can _run_ a variable by calling the function `.run()`. In the case of a fixed variable, it will set its value in `.value`. 

# %%
counts.value

# %%
counts.run()

# %% [markdown]
# The value can now be accessed, and because we're working with torch, this value is a `torch.tensor`:

# %%
counts.value

# %% [markdown]
# Because our variable is annotated, we can also get the same output as a data frame:

# %%
counts.value_pd.head()

# %% [markdown]
# We can also provide pandas and xarray objects to {class}`latenta.variables.Fixed`, and the definition of a variable (its dimensional structure) will directly be inferred from the object's indices (if we gave them proper names beforehands).
#
# Let's try with the level of overexpression in our data:

# %%
adata.obs["log_overexpression"]

# %%
overexpression = la.Fixed(adata.obs["log_overexpression"], label="overexpression")
overexpression

# %% [markdown]
# As we can see, we did not provide a definition to this variable, but it was inferred from the series provided based on its index name (`adata.obs["log_overexpression"].index.name`, i.e. 'cell'). 
#
# The first dimensions of both our `counts` [cells, genes] and `overexpression` [cells] are therefore equal:

# %%
counts[0] == overexpression[0]

# %% [markdown]
# Regarding **discrete fixed variables**, we often like to work in a "one-hot" encoding, meaning that we get a binary (True/False or 1/0) matrix if a sample (cell) is part of a particular group. `la.variables.DiscreteFixed()` performs the conversion of a categorical pandas series to one-hot encoding if necessary. 
#
# In our data, an example of a discrete fixed variable is the variable indicating what (*Myod1* or control mCherry) is overexpressed in each cell:

# %% [markdown]
# ```{margin}
# A pandas categorical series is equivalent to R's factor
# ```

# %% tags=[]
adata.obs["gene_overexpressed"]

# %% [markdown]
# Let's convert it to a discrete fixed variable called `overexpressed` using `la.variables.DiscreteFixed`

# %%
overexpressed = la.variables.DiscreteFixed(adata.obs["gene_overexpressed"])
overexpressed


# %%
overexpressed.run()
overexpressed.value_pd.head()

# %% [markdown]
# ## 2. Parameters

# %% [markdown]
# Parameters are unknown variables that have to be inferred (optimized) based on our data. They require a starting default value. A typical example is the slope of the genes, or basically their log fold-change, in our specific case it will represent how much the expression of a gene changes with the overexpression of *Myod1*. 

# %%
slope = la.Parameter(
    0.0, definition=la.Definition([genes]), label="slope", symbol=r"\delta"
)
slope

# %% [markdown]
# Another parameter is the baseline expression of the genes, i.e. how much a gene is expressed by default when *Myod1* is not overexpressed. Let's initialize it at 1:

# %%
baseline = la.Parameter(
    1.0,
    definition=la.Definition([genes]),
    label="baseline",
    transforms=[la.transforms.Exp()],
)
baseline

# %% [markdown]
# You might have noted that we add a transformation to the baseline by specifying 
# `transforms = [la.transforms.Exp()]`.
#
# Indeed, although parameters are _free_, not fixed, they can still have constraints. For example here, the baseline expression of a gene can never go below zero, as this would be non-sensical. However, most algorithms cannot directly cope with these constraints and prefer to work with parameters that can go from $-\infty$ to $+\infty$ (especially if you also want flexibility). We can solve this by transforming each parameter to make sure they fit our constraints.

# %% [markdown]
# Frequently used transformations are:
#
# Description | Support | Transform | Formula
# --- | --- | --- | ----
# All positive numbers | $R^+$ | `.Exp()` | $e^x$
# Unit interval | $[0, 1]$ | `.Logistic()` | $\frac{1}{1+e^{-x}}$
# Circular (i.e. an angle) | $[0, 2\pi[$ | `.Circular()` | $ atan2(y, x) $
# Simplex | $\in [0, 1] \wedge \sum = 1$ | `.Softmax()` | $\frac{e^{x_i}}{\sum_i e^{x_i}}$

# %% [markdown]
# A combination of fixed and parameter variables form the leaves of our model. Any other types of variables in our models are ultimately constructed from these leaf variables.

# %% [markdown]
# ## 3. Computed variables

# %% [markdown]
# The leaf variables are used to compute unknown information about our data or to be able to generalize certain results to any future cells. In this case, we create a variable that will depend on these other variables, which we then call components. This is done most of the time thanks to a regression model and its corresponding link function to modify the input data (see part 2. of the tutorial).
#
# Let's take the example of the gene expression which we want to model. In our data for example, we hypothetise and are interested in how the overexpression of *Myod1* (`overexpression`) affect the gene expression. The gene expression will be modified by a certain factor (`slope`) that will be specific to each gene.

# %% tags=["hide-input"]
import matplotlib.pyplot as plt

with plt.xkcd():
    np.random.seed(1234)
    x = np.random.uniform(0, 2, 18)
    y = x + np.random.uniform(-0.4, 0.4, 18)
    plt.scatter(x, y)
    plt.plot(np.linspace(0, 2, 9), np.linspace(0, 2, 9))
    plt.hlines(0.75, 0.75, 1, colors="gray")
    plt.vlines(1, 0.75, 1, colors="gray")
    plt.text(x=1.06, y=0.80, s="slope", backgroundcolor="lightgray")
    plt.ylabel("expression of gene g")
    plt.xlabel("overexpression of Myod1")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.show()

# %% [markdown]
# How much each gene is expressed in each specific cell will then depend on how much it overexpresses *Myod1* and we end up with the formula:
# `expression`$_{g,c}$ ~ `slope`$_{g}$*`overexpression`$_{c}$ for a gene $g$ and a cell $c$
#
# The function `la.links.scalar.Linear` will directly model this for us:

# %%
expression = la.links.scalar.Linear(x=overexpression, a=slope, label="expression")
expression

# %% [markdown]
# The two dimensions of this variable are **broadcasted**, i.e. their coordinates and size are set by upstream components (signified by a •).
#
# The actual definition of the variable can be obtained using `.value_definition`:

# %%
expression.value_definition

# %% [markdown]
# When building models, and especially complex models, it gets  really handy to see how the different variables are related in a graph structure:

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
# Often, components can still be added or changed after the initialisation of a variable. For example, we can now add an intercept (`baseline` defined earlier) to `expression` such that `expression`$_{g,c}$ ~ `slope`$_{g}$*`overexpression`$_{c}$ + `baseline`$_{g}$

# %%
expression.a

# %%
expression.b = baseline

# %% tags=[]
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
# ## 4. Modular variables

# %% [markdown]
# Sometimes, a computed variable can depend on many variables, and it can become handy to be able to link them easily.
#
# For example, for now the gene `expression` is "simply" modeled based on a baseline expression and the overexpression of *Myod1*, however it could also include the batch from which each cell comes from, or the phase of the cell cycle in which the cell is, etc. It would of course be possible to use the function `la.links.scalar.Linear` to compute the expression as a combination of all these variables, but it would be quite convoluted. Thankfully, instead we can use `la.modular`!
#
# Let's try to redefine `expression` as a modular variable, and as we want to use an additive (linear) model we will use `la.modular.Additive`.

# %%
expression = la.modular.Additive(definition=[cells, genes], label="expression")

# %%
expression

# %% [markdown] slideshow={"slide_type": "notes"} tags=[]
# Now we have our modular variable `expression` to which we can add the different variables that will be automatically combined in a linear fashion.
# Note that we are not anymore confined to the usual components defined previously, we can add as many as we want, and called them as we want.

# %%
expression.baseline = baseline
expression.overexpression = la.links.scalar.Linear(overexpression, a=slope)

# %%
expression.overexpression

# %%
expression.plot()

# %% [markdown]
# We can see that the two arguments of `expression`, `baseline` and `overexpression`, were automatically combined. Imagine we want to add the batch effect for example, it would be really easy to do so by adding a new argument.

# %% [markdown]
# ## 5. Distributions

# %% [markdown]
# We are often uncertain about the exact value of some variables. For example, for a certain cell and gene we might found that the `expression` value is 4, but we also know that data are noisy and we can therefore not be completly sure that the expression value is exaclty 4, we rather think that it can be somewhere around 4. To model this uncertainty we would use a distribution.

# %% [markdown]
# There are two ways to initialize a distribution variable:
# - We can take a random sample: Let's imagine a simple (non-biological) example, we have a die, we roll it, and get 3.
# - We have some information, and we can calculate the probability of a particular sample, also known as the likelihood: For a fair die, the probability of observing a 3 is 1/6.

# %% [markdown] tags=["remove-output", "hide-input"]
# It can be important to keep in mind that depending on what we want to model as a distribution, individual elements of a sample can be dependent. Furthermore, this dependence can be specific to particular dimensions. For example, if we sample from an OneHotCategorical distribution, there can only be one 1 in every row.

# %% [markdown]
# Let's continue with the example of gene expression. We previously defined the computed variable `expression`, however as we just mentionned, we cannot be certain that the expression will be these exact values, but it will rather range around this value. A distribution we typically use for gene expression is a negative binomial, and more specifically, the `NegativeBinomial2` variant that has two input parameters: the average count (`.mu`) and the dispersion (`.dispersion`). (Note that we denote the fact that it is a distribution variable with _p)

# %% [markdown]
# ```{margin}
# The naming conventions used are those from ProbOnto (https://sites.google.com/site/probonto/home) whenever available.
# ```

# %%
transcriptome_p = la.distributions.NegativeBinomial2(mu=expression)

# %% [markdown]
# Note that most commonly used distributions have an _average_, also called a location, loc, mu or mean, and also an _uncertainty_, also called the standard deviation, sigma, scale or dispersion. This is also the case for the `NegativeBinomial2`, for which we did not specify the `dispersion`, and is then by default set to 1.

# %% [markdown]
# Running a distribution will pick a certain value from the distribution:

# %%
transcriptome_p.run()
transcriptome_p.value_pd.head()

# %% [markdown]
# Because we model this variable as a distribution, we can calculate how likely the exact values we just picked are according to the distribution. The goal of any modeling is to maximize this likelihood while remaining within the constraints imposed by the model.

# %%
transcriptome_p.likelihood

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
# ## 6. Observations

# %% [markdown]
# Observations are fixed variables that we observe and follow a distribution since we know the observed values are uncertain. An observation always has some physical connection to reality, but the problem is that we only observe a noisy variant of this reality (the observed `counts` are one noisy version of the "real" transcriptome). The noise is not always purely technical (due to stochastic sampling of mRNAs) but also typically contains biological components (such as transcriptional bursting). Essentially, any variation that the model cannot explain is explained away as noise. In our case, this noise is in the dispersion component of the Negative Binomial.

# %%
transcriptome = la.Observation(
    value=counts.value, p=transcriptome_p, definition=counts_definition
)

transcriptome.plot()

# %% [markdown]
# Running an observation does the same thing as running a fixed variable, meaning it sets its values to the values we provided (i.e. `value = counts.value`)

# %%
transcriptome.run()

# %% [markdown]
# Again, because we model an observation as a distribution, we can calculate how likely this precise  observation is according to its distribution. 
#
# The goal of modelling is to maximize this likelihood, while remaining within the constraints imposed by the model. All along the modelisation process the parameters of the distribution of the transcriptome `transcriptome_p` will be modified/optimized to find the most likely set of parameters given our input, i.e the `counts.value`. 

# %%
transcriptome.likelihood

# %% [markdown]
# ## 7. Latent variables

# %% [markdown]
# Last but not least: latent variables are, by definition, variables we believe exist but are not directly observable through experimentation. They are, therefore, unknown variables that the model needs to estimate, exactly like parameters. However contrary to parameters which have one fixed value, latent variables follow a distribution, and who says distribution says modelisation of uncertainty! 
#
# As commonly done, Latenta uses Bayesian statistics to infer latent variables. Don't worry if you are not familiar with Bayesian statistics, you do not need to. Here are the main ideas behind it (if you are interested to know more we suggest to have a look at THIS).  
#
# The main principle behind Bayesian modelling is to model parameters as a distribution. For that we first need to define what is called the prior, what we think the variable will look like before looking at the data. For example, I believe that most of my genes will not be affected by the overexpression, i.e the distribution of fold change will be centered in 0. Then this prior knowledge is updated by what we observe in the data (i.e for gene A indeed the foldchange is very low, but for gene B it seems to be very high). This will give us what is called the posterior distribution. This represent the distribution of the variable now that we observed the data. 
#
# This Bayesian modelling approach adds a crucial statistical layer as it permits to: 
# * put some constrains on the model (mainly with the prior) 
# * assess statistical significance, and repport how certain we are of our results (Bayesian statistics)
#     This is true at the parameter level but also to compare entire models between each others
# * maintain uncertainty, we can model the uncertainty over the parameters and also to model the noise in the data (since we now use distributions).  
#
# Latent variables are central to probabilistic modelling, because they encompass two types of uncertainty:
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
# In general, we call this _aleatoric uncertainty_, it refers to the noise inherent in the data, it is the uncertainty caused by **randomness**. In bayesian modelling, this type of uncertainty is typically encoded as the prior distribution. 
# :::
# ::::
#
# Let's say we are randomly sampling a cell from a tissue. Every time we take such a sample, we do not know what type of cell it will be, except perhaps that some cell types are more likely to be picked because they are more abundant. This uncertainty is inherent to the population, and nothing we do can change that. Or in the same logic if we could sample many times the transcriptome from the same cell, even at the exact same time, we would each time get a slightly different output. 
# We model this uncertainty as an appropriate probability distribution, which can have some known or unknown components. 
#
# This type of uncertainty is often of interest and can provide interesting biological information in more complex models.
#
# For example:
# - How does cell type abundance change across different conditions?
# - What is the distribution of all cell's pseudotime? Are there more early than late cells? Does this change between conditions?
# - What is the distribution of the gene's fold changes? Are there more upregulated than downregulated genes? Are there a couple of genes with massive changes, while all other genes do not change at all?
# - What is the effect of a transcription factor (like *Myod1" in our example)? Does it have many target genes with a subtle effect? Or a few target genes but with a very strong effect?

# %% [markdown]
# ### 2. Uncertainty because of lack of data
#
# ::::{margin}
# :::{note}
# In general, we call this _epistemic uncertainty_, it refers to the uncertainty in the model which could be explained given enough data, it is the uncertainty caused by **ignorance**. In bayesian modelling, this type of uncertainty is typically encoded as the posterior.::: ::::
#
# Let's say we focus on one particular cell, and we want to know its cell type. Naturally, before we observe anything about this cell, our uncertainty is the same as that inherent to the system (as described above). However, if we now observe some gene expression, our uncertainty for this particular cell decreases. The more genes we observe, the more certain we will become.
#
# This type of uncertainty is not inherent to the cell. The cell does not change its cell type just because we are uncertain about it. Therefore it does not have a direct connection to reality but is simply an artefact of us not knowing enough. Nonetheless, modelling this uncertainty is crucial because it gives us an idea about how certain we are about a variable given our data. For example:
# - We may be very sure about a cell's cell type but be very uncertain about the cellular state. This could tell us that we do not have enough sequencing depth to assign a cell's state.
# - If a gene's fold-change has a distribution spreading over both negative and positive we might not have enough data. This does not necessarily mean that the gene is not differentially expressed, but rather that we do not have enough data to know whether it is.
# - It may be that two transcription factors can equally regulate gene expression in the model and that we do not have enough data to say which one is more likely.

# %% [markdown]
# Don't worry if it feels difficult to grasp the exact differences between the two types of uncertainty and why they are modeled by the prior or posterior. While being interesting, it is not a necessary concept to understand in order to properly constructs models, the important part is to understand the concept of uncertainty as a whole.

# %% [markdown]
# ### Constructing a latent variable
#
# Latent variables combine both types of uncertainty. 

# %% [markdown]
# Name | Component | Symbol | Description
# --- | --- | --- | ----
# Prior distribution | `.p` | $p$ | The distribution followed by the variable, inherent to the system. Note that this distribution can depend on other parameters, latent or fixed variables.
# Variational distribution | `.q` | $q$ | An approximation of the posterior distribution, i.e. the distribution followed by a variable after observing the data. The parameters of this distribution can be estimated directly (i.e. as one or more Parameter) or through the observed data using amortization
#
# Basically, any free parameter can be modeled as a latent variable. If we look at our graphical model, the baseline and the slope could both be defined as latent variables. Let's start with the slope. 
#
# It is a reasonable assumption that most gene will not be very affected by the overexpression of *Myod1*. Therefore, we want to add some "prior" knowledge/constraints, using the prior distribution $p$, to not allow the model to find/fit enormous fold-changes, except if the data provides strong evidence that it is indeed the case (we will discuss further in the regression part). Let's now redefine the slope as a latent variable:

# %%
slope_p = la.distributions.Normal(
    loc=0.0, scale=la.Parameter(1.0, transforms=[la.transforms.Exp()])
)
latent_slope = la.Latent(p=slope_p, definition=la.Definition([genes]), label="slope")
latent_slope.plot()


# %% [markdown]
# The prior distribution $p$, in this case, is a normal distribution centered in 0 with one free parameter: the scale $\sigma$. This parameter determines how far the slope _on average_ can be different from 0. We can estimate the scale as a parameter only because we are pooling information across many genes, otherwise it would be more judicious to set it at a fix value. This kind of {term}`multi-level modelling` is very powerful, as it includes multiple testing correction directly within the model {citel}`gelman_why_2009`.
#
#
# The posterior distribution, or more precisely the surrogate/approximation of the posterior distribution, $q$, is by default set as being a normal distribution too. However, it contains two parameters specific for each gene. These contain our idea of where the slope of each gene will lie: the location $\mu$ is the average, while the scale $\sigma$ our uncertainty.

# %% [markdown]
# We can now modify the slope (`.a`) of the computed variable `expression` with a latent variable rather than a parameter.

# %%
expression.a = latent_slope
expression.plot()

# %% [markdown]
# Note that many link functions will create a latent variable automatically if you specify `True` for the coresponding argument, although you have to provide the correct definition if some dimensions cannot be inferred from other components. For example for the `expression` we could have directly done:

# %%
expression = la.links.scalar.Linear(
    overexpression,
    a=True,  # 👈
    b=baseline,
    label="expression",
    definition=la.Definition([cells, genes]),  # ⚠️
)
expression.plot()

# %%
expression.a.p

# %% [markdown]
# :::{note}
# Sometimes components of the prior distribution may themselves depend on a latent variable. In that case, this distribution will encompass both types of uncertainty. This would be the case for [most examples we gave for prior distributions](#uncertainty-inherent-to-the-system).
# :::

# %% [markdown]
# Just like parameters, the two distributions of latent variables may also have transformations. This is, for example, the case if we would use a `.LogNormal` as prior, which only takes positive numbers as input. A good example is the baseline expression of the genes:

# %%
baseline_p = la.distributions.LogNormal(
    loc=la.Parameter(0.0), scale=la.Parameter(1.0, transforms=[la.transforms.Exp()])
)
latent_baseline = la.Latent(
    p=baseline_p, definition=la.Definition([genes]), label="baseline"
)
latent_baseline.plot()

# %% [markdown]
# We can now modify the baseline (`.b`) of expression as a latent variable:

# %%
expression.b = latent_baseline
expression.plot()

# %% [markdown] tags=["remove-output", "hide-input"]
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
# - Combining several effects is possible using modular variables
# - Distributions model uncertainty, and often (but not always) have components that model the mean and variability
# - Latent variables are the variables we are the most interested in, as they encompass both the uncertainty inherent to the system and the uncertainty due to the lack of data
#
# Now that we know how variables work, let's look at how we can combine these different variables to model and find optimal values for the free parameters in the next tutorial

# %%
