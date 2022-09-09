# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Leaf variables

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import latenta as la
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import torch

# %%
cells = la.Dim(10000, "cell")
genes = la.Dim(1000, "gene")
definition = la.Definition([cells, genes])

# %%
import tempfile

# %%
tmpdir = tempfile.TemporaryDirectory().name
pathlib.Path(tmpdir).mkdir()

# %%
filesize

# %% [markdown]
# ## Loaders

# %%
value = np.zeros(definition.shape)
for i in range(value.shape[0]):
    value[i, np.random.choice(value.shape[1])] = 1.0

# %%
value.sum()

# %% [markdown]
# ### Memory loader

# %%
loader = la.loaders.MemoryLoader(value)
fixed = la.Fixed(loader, definition=la.Definition([cells, genes]))

# %% [markdown]
# ### Sparse loader

# %% tags=["hide-input"]
import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
            if torch.is_tensor(obj):
                size += obj.element_size() * obj.nelement()
        objects = get_referents(*need_referents)
    return size


# %% [markdown]
# Obviously, a large difference in memory usage.

# %%
getsize(loader) / 1024**2

# %% [markdown]
# Using 3GB (or more) on current GPUs would be prohibitive.

# %%
sparse = la.sparse.COOMatrix.from_numpy_array(value)
loader_sparse = la.loaders.SparseMemoryLoader(
    sparse
)
fixed_sparse = la.Fixed(loader_sparse, definition=la.Definition([cells, genes]))

# %%
loader_sparse.initialize(fixed_sparse)

# %%
getsize(loader_sparse) / 1024**2

# %% [markdown]
# #### Initialization

# %% [markdown]
# There are several ways to initialize the COOMatrix:

# %%
la.sparse.COOMatrix.from_numpy_array(value)

# %%
value_scipy_coo = scipy.sparse.coo_matrix(value)
la.sparse.COOMatrix.from_scipy_coo(value_scipy_coo)

# %%
value_scipy_csr = scipy.sparse.csr_matrix(value)
la.sparse.COOMatrix.from_scipy_csr(value_scipy_csr)

# %% [markdown]
# #### Timing

# %% [markdown]
# When data is accessed from a SparseMemoryLoader, it is converted to a dense tensor. This can cause a slowdown:

# %%
import timeit

# %%
time = timeit.timeit("loader.get()", number=1, globals=globals())
time

# %%
time = timeit.timeit("loader_sparse.get()", number=1, globals=globals())
time

# %% [markdown]
# Speed is however much more competitive when taking only a subsample, which is the typical use case anyway when working with large sparse tensors:

# %%
idx = {"cell": torch.tensor(range(500))}

# %%
time = timeit.timeit("loader.get(idx)", number=1, globals=globals())
time

# %%
time = timeit.timeit("loader_sparse.get(idx)", number=1, globals=globals())
time

# %% [markdown]
# ### File loader

# %% [markdown]
# Supply a path to a pickle. The value will not be saved along with the loader

# %%
import pickle
import tempfile
import pathlib
import os

# %%
path = pathlib.Path(tempfile.NamedTemporaryFile().name)
pickle.dump(value, path.open("wb"))

# %%
filesize = os.path.getsize(path) / 1024**2
print(filesize, "Mb")

# %%
assert filesize > 1024

# %%
fixed = la.Fixed(path, definition = la.Definition([cells, genes]))

# %%
fixed_path = pathlib.Path(tempfile.NamedTemporaryFile().name)
pickle.dump(fixed, fixed_path.open("wb"))

# %%
filesize = os.path.getsize(fixed_path) / 1024**2
print(filesize, "Mb")

# %%
assert filesize < 3

# %% [markdown]
# ### Sparse File Loader

# %% [markdown]
# Similarly, we can load a sparse matrix from a file. This is especially useful with larger tensors, i.e. over a billion values (e.g. hundred thousands cells and ten thousand genes)

# %%
path = pathlib.Path(tempfile.NamedTemporaryFile().name)
pickle.dump(sparse, path.open("wb"))

# %%
filesize = os.path.getsize(path) / 1024**2
print(filesize, "Mb")

# %%
assert filesize > 3

# %%
fixed = la.Fixed(
    la.loaders.COOMatrixFileLoader(path),
    definition = la.Definition([cells, genes])
)

# %%
fixed_path = pathlib.Path(tempfile.NamedTemporaryFile().name)
pickle.dump(fixed, fixed_path.open("wb"))

# %%
filesize = os.path.getsize(fixed_path) / 1024**2
print(filesize, "Mb")

# %%
assert filesize < 3

# %%
fixed2 = pickle.load(fixed_path.open("rb"))

# %%
fixed2.loader.get()

# %%
