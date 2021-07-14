---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

# Part 1: Modelling

```python
import latenta as la
import scanpy as sc
```

```python
adata = sc.datasets.pbmc3k()
```

Modelling means explaining what we observe using things we want to know. For example, we may be interested in how a cell differentiates, and thus create a model in which we try to explain the transcriptomes of many differentiating cells by fitting a model of gene dynamics along differentiation.


Creating and inferring a model consists of several steps:
1. Creating known and unknown variables
2. Connect these variables, i.e. creating the model
3. Infer the value of the unknown variables
4. Interpret the model
5. Go back to step 1, and compare different competing models

In this tutorial, we go briefly over each step. More details on using latenta to interpret specific types of datasets can be found in the [tutorials](/tutorials), while detailed explanations on specific problems (e.g. the cell cycle) can be found in the [user guide](/guide).


## Variables


### Variable definition


A variable in latenta is a tensor, meaning it is a set of numbers that live in zero (scalar), one (vector), two (matrix) or more dimensions. Each dimension is annotated with a label, symbol and/or description, and can have further annotation.

For example, a dimension can have as label "gene", symbol $g$:

```python
genes = la.Dim(adata.var.index, id = "gene", symbol = "g")
genes
```

Similarly, for cells:

```python
cells = la.Dim(adata.obs.index, id = "cell", symbol = "c")
cells
```

We can define how a variable should look like, without any data, using `la.Definition()`:

```python
counts_definition = la.Definition(
    [cells, genes],
    "counts"
)
counts_definition
```

### Fixed variables


Fixed variables never change.

```python
la.Fixed(adata.)
```

### Parameters


Parameters are variables that are unknown and have to be inferred (in our case, optimized) based on the data. Parameters do require a starting default value.

```python
la.Parameter()
```

A combination of fixed and parameter variables form the leaves of our model. In other words, any other types of variables in our models are ultimatily constructed from these leaf variables.


### Distributions


We often are uncertain about the value of some variables, and this uncertainty is encoded as a distribution. A distribution has two characteristics:
- We can take a random sample
- We can calculate the probability of a particular sample, also known as the likelihood

Both characteristics are used during inference, interpretation and data generation.


A lot of commonly used distributions have an "average", also called a location, loc, mu or mean, and an "uncertainty", also called the standard deviation, sigma, scale or dispersion.


Depending on the distribution, individual elements of a sample can be dependent. This dependence can be specific to particular dimensions.


There are two types of uncertainty:
- Uncertainty inherent to the system. If we take a random cell from a population, this cell will be in a some unknown position in the cell cycle. This uncertainty is simply inherent to the population of cells. There's nothing we can do to reduce this uncertainty in this population.
- Uncertainty because of the lack of data. If we take a random cell but only observe a couple of genes, we will be quite uncertain about its cell cycle position. However, we can become more certain by observing more genes of this particular cell.

The first type of uncertainty we want to model and is often of interest to us as it tells us something about our system. The second type of uncertainty we want to reduce as much as possible in time, but we still need to model it because we are limited in data size (e.g. number of biological replicates).


### Observations


Observations are fixed variables that follow a distribution. An observation always has some physical connection to reality, but the problem is that we only observe a noisy variant of this reality. This noise is not purely technical (due to stochastic sampling of mRNAs) but also typically contains biological components (such as transcriptional bursting). Essentially, any variation that cannot be explained by the model is explained away as noise.


Because we model an observation as a distribution, we can calculate how likely this observation is according to this distribution. The goal of any modelling is to maximize this likelihood, while remaining within the constraints imposed by the model. And the only way to maximize this likelihood is by tuning parameters upstream of the observation's distribution.


Let's illustrate this with a first simple model. We have one fixed variable "time", and one parameter "slope". 

```python

```
