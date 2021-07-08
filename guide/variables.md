```python
%load_ext autoreload
%autoreload 2

import latenta as la
import numpy as np
import pandas as pd
```

## Linear regression


### Generate some data

```python
n_cells = 100

x_value = pd.Series(np.random.uniform())
```

## Definition



A variable in latenta is a tensor, which has zero or more discrete dimensions (e.g. cells, genes, ...) and one continuous dimensions (e.g. gene expression). This continuous dimension also has a domain associated with it, such as the Real numbers, positive numbers, ...

```python
dim = la.DimFixed(pd.Series(['a', 'b'], name = "letters"))
dim = la.DimFixed(['a', 'b'], id = "letters")
```

```python
dim
```

## Variables


In latenta, we build up a directed graph of how variables are related to eachother.

This graph is often a directed acyclic graph, although some parts of the graph can be cyclic.

The root of the graph are typically real-life observations, with the goal of building a model that best explains what we observe.

<!-- #region -->





Each variable in latenta is annotated. This means that the variable as a whole has a label, symbol and/or description, while each dimensions has a label and associated metadata.



## Variable

## Leafs

### Fixed

### Parameters

## Random

## Distributions

## Computed

- Definition: Annotated tensor
  - Variable: An annotated tensor within a DAG, with components and children
    - Leaf: A variable without any components.
      - Parameter: A leaf that can be optimized.
      - Fixed: A leaf that has a fixed (either observed or given) value.
    - Random: A random variable, i.e. follows a certain distribution.
      - Observed: A random variable that is observed.
      - Latent: A random variable that needs to be inferred.
    - Distribution: A variable that is sampled from a distribution.
    - Computed: A variable that can be determinstically computed from other variables.
      - Transform: An invertible transform of a variable. Does not have to be bijective.
      - Modular: A variable that sums or multiplies different variables. One dimension can be subsetted.

### Transforms

### Modular

All other classes extend from these types of variables.
<!-- #endregion -->

```python

```
