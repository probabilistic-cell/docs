---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=[] -->
# Doublets 😎🦒
<!-- #endregion -->

```python
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import torch

import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import collections

import latenta as la

la.logger.setLevel("INFO")
```

## Model 1

```python
n_cells = 50
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim("C" + pd.Series(cell_ids, name="cell"))

n_genes = 100
genes = la.Dim("G" + pd.Series([str(i) for i in range(n_genes)]), name="gene")

celltype_ids = "Ct" + pd.Series(["a", "b"])
celltypes = la.Dim(celltype_ids, name = "celltype")
```

```python
# celltype_value = pd.Series(np.random.choice(["a", "b"], n_cells, replace = True), index = cells.index)
```

```python
baseline_expression_value = pd.DataFrame(
    np.random.normal(scale = 3, size = (celltypes.size, genes.size)), index = celltypes.index, columns = genes.index,
)
baseline_expression = la.Fixed(baseline_expression_value, label = "baseline")
```

```python
proportions_value = pd.DataFrame(np.random.dirichlet((3., 5), size = (cells.size)), index = cells.index, columns = celltypes.index)
# proportions_value = pd.DataFrame(np.random.dirichlet((1., 1.), size = (cells.size)), index = cells.index, columns = celltypes.index)
```

```python
# proportions_value = np.exp(proportions_value)/np.exp(proportions_value).sum(1).values[:, None]
sns.heatmap(proportions_value)
```

```python
proportions = la.Fixed(proportions_value, label = "proportions")
```

```python
expression = la.links.vector.Matmul(proportions, baseline_expression, definition = la.Definition([cells, genes]), label = "expression")
```

```python
expression.plot()
```

```python
noise = la.Fixed(0.0001)
```

```python
dist = la.distributions.Normal(expression, noise, label = "dist")
```

```python
dist.plot()
```

```python
posterior = la.posterior.Posterior(dist)
posterior.sample(1)
```


```python
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0) = plt.subplots(1, 1, figsize=(5, 5))
cell_order = proportions_value[celltype_ids[0]].sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)
```

```python
sns.scatterplot(proportions_value.loc[cell_order, celltype_ids[0]], observation_value.loc[cell_order]["G2"])
```

```python
gene_ids_oi = genes.coords[:5]

fig, axes = la.plotting.axes_wrap(len(gene_ids_oi))

for gene_id, ax in zip(gene_ids_oi, axes):
    # print(gene_id, )
    # print(baseline_modifier.prior_pd()[gene_id])
    sns.scatterplot(
        x = proportions_value.loc[cell_order, celltype_ids[0]],
        y = observation_value.loc[cell_order][gene_id],
        ax = ax
    )
    for celltype in celltype_ids:
        ax.axhline(baseline_expression_value.loc[celltype, gene_id], label = celltype)
plt.legend()
```

## Model 2

```python
n_cells = 50
cell_ids = [str(i) for i in range(n_cells)]
cells = la.Dim("C" + pd.Series(cell_ids, name="cell"))

n_genes = 100
genes = la.Dim("G" + pd.Series([str(i) for i in range(n_genes)]), name="gene")

celltype_ids = "Ct" + pd.Series(["a", "b"])
celltypes = la.Dim(celltype_ids, name = "celltype")
```

```python
baseline_expression_value = pd.DataFrame(np.random.normal(scale = 3, size = (celltypes.size, genes.size)), index = celltypes.index, columns = genes.index)
baseline_expression = la.Fixed(baseline_expression_value, label = "baseline_original")
```

```python
baseline_modifier_value = (np.random.random(baseline_expression.shape) > 0.5) * (np.random.choice([-1, 1], baseline_expression.shape)) * (np.random.normal(3.0, 1.0, baseline_expression.shape))
baseline_modifier = la.Fixed(baseline_modifier_value, definition = baseline_expression, label = "modifier")
baseline_modifier_value = baseline_modifier.prior_pd()
```

```python
responded_baseline_expression = la.links.scalar.Linear(baseline_expression, b = baseline_modifier, label = "baseline")
```

```python
proportions_value = pd.DataFrame(np.random.dirichlet((3., 5), size = (cells.size)), index = cells.index, columns = celltypes.index)
```

```python
proportions = la.Fixed(proportions_value, label = "proportions")
```

```python
# expression = la.links.vector.Matmul(proportions, baseline_expression, output = la.Definition([cells, genes]), label = "expression")
expression = la.links.vector.Matmul(proportions, responded_baseline_expression, definition = la.Definition([cells, genes]), label = "expression")
```

```python
expression.plot()
```

```python
noise = la.Fixed(0.1)
```

```python
dist = la.distributions.Normal(expression, noise, label = "dist")
```

```python
dist.plot()
```

```python
posterior = la.posterior.Posterior(dist)
posterior.sample(1)
```


```python
observation_value = posterior.samples[dist].sel(sample=0).to_pandas()
fig, (ax0) = plt.subplots(1, 1, figsize=(5, 5))
cell_order = proportions_value[celltype_ids[0]].sort_values().index
sns.heatmap(observation_value.loc[cell_order], ax=ax0)
```

```python
gene_ids_oi = genes.coords[:5]

fig, axes = la.plotting.axes_wrap(len(gene_ids_oi))

for gene_id, ax in zip(gene_ids_oi, axes):
    # print(gene_id, )
    # print(baseline_modifier.prior_pd()[gene_id])
    sns.scatterplot(
        x = proportions_value.loc[cell_order, celltype_ids[0]],
        y = observation_value.loc[cell_order][gene_id],
        ax = ax
    )
    for color, celltype in zip(["red", "blue"], celltype_ids):
        ax.axhline(baseline_expression_value.loc[celltype, gene_id], label = celltype, color = color)
        ax.axhline(baseline_expression_value.loc[celltype, gene_id] + baseline_modifier_value.loc[celltype, gene_id], label = celltype, color = color, dashes = (2, 2))
plt.legend()
```

```python

```

```python

```
