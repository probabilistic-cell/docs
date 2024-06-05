# -*- coding: utf-8 -*-
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
# # Workflows

# %% [markdown]
# For now we mainly looked at models in isolation. However, the true power of probabilistic modelling lies in its ability to move beyond the "one-model-fits-all" idea, but rather create a variety of models and connect or compare them in various ways. Indeed, even for a normal use case, we already typically go up to a dozen models or more. In more complex experimental designs, which are becoming every more common, this can easily go up to a hundred or more.

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac
import numpy as np

# %% [markdown]
# Even in relatively simple use cases, managing different models, storing them across sessions, and let them depend on each other can be a pain. To make this easier, we created a small workflow manager in the _laflow_ package. It borrows elements from other workflow managers such as snakemake or nextflow. However, it is much more focused on interactivity, modularization and model inheritance, all concepts very central to the modelling we're doing here.
#
# The core tasks of laflow is:
# - **Persistence**, to store models and its derivatives on disk
# - **Model inheritance**, creation of similar models that only differ in certain aspects
# - **Transfer models**, to transfer (sub)models to reuse with other models

# %%
import laflow as laf

# %% [markdown]
# We'll use the same dataset as [before](./1-variables).

# %%
adata = la.data.load_myod1()

# %% tags=["hide-input", "remove-output"]
import scanpy as sc

adata.raw = adata

sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

sc.pp.combat(adata)
sc.pp.pca(adata)

sc.pp.neighbors(adata)
sc.tl.umap(adata)

adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])

# %% [markdown]
# ## Persistence

# %% [markdown]
# The basis of _laflow_ is the {class}`~laflow.Flow` class. Each instance we create is associated with a particular folder on disk, starting from a given root directory. We work from a temporary directory in these tutorials, but in real life use cases this root is typically  your project's root directory or one of its subdirectories.

# %%
import tempfile
import pathlib

project_root = pathlib.Path(tempfile.TemporaryDirectory().name)
project_root.mkdir()
laf.set_project_root(project_root)  # sets the default project root

# %% [markdown]
# We can then create a {class}`~laflow.Flow` object as follows. Because we have set the default root for this python session, we don't have to provide this explicitely. All we need to provide is a name for the flow:

# %%
dataset = laf.Flow("dataset")
dataset.path  # the path associated with this flow

# %% [markdown]
# This folder will store information on each object, links to other objects or flows, and the objects themselves if they are labeled as persistent. We can store a python object inside a flow by simply assigning it:

# %%
dataset.adata = adata

# %% [markdown]
# Because python objects are by default persistent, the `adata` object was saved in the folder as a pickle. Along with it is a info json file which is used internally to keep track which type of object it is and how it was created.

# %%
# !ls -lh {dataset.path}

# %% [markdown]
# When working in IPython environments (e.g. Jupyter), we can always check what contents a flow object has as follows:

# %%
dataset

# %% [markdown]
# Even if we restart our session (simulated here by removing the flow object), we can always retrieve it by recreating a flow object in the correct folder:

# %%
if "dataset" in globals():
    del dataset
dataset = laf.Flow("dataset")

# %%
dataset

# %% [markdown]
# We can also have persistence of other objects. If you provide a matplotlib object, this will be stored as a png and pdf:

# %%
# don't forget the return_fig, or this function will return None and that will be stored as a pickle
dataset.basic_umap = sc.pl.umap(dataset.adata, return_fig=True)

# %%
dataset

# %%
# !ls -lh {dataset.path}

# %% [markdown]
# ## Creating models

# %% [markdown]
# `laflow` already contains a `laf.Model` class that contains the basic things necessary to construct and infer a model

# %%
model = laf.Model("model")

# %%
model.root_initial = la.Root(
    p = la.Latent(p = la.distributions.Normal(), definition = [la.Dim(range(100), "a")])
)

# %%
model.infer()

# %%
model.trace.plot()


# %% [markdown]
# ## Connecting flows

# %% [markdown]
# To make our workflows more modular, we use Python's OOP system to systematically let certain flows depend on each other and share common code. For example, 

# %%
class Dataset(laf.Flow):
    default_name = "dataset"
    adata = laf.PyObj()


# %% [markdown]
# We can initialize this dataset as before. Note that we do not provide a name in this case as we want to use the default "dataset" name:

# %%
dataset = Dataset(adata=adata)

# %%
dataset


# %% [markdown]
# If we would then make a model, we could make it depend on the dataset as follows:

# %%
class Model(laf.Flow):
    default_name = "model"
    dataset = laf.FlowObj()


# %%
model = Model(dataset=dataset)

# %%
model

# %% [markdown]
# The dataset is now linked to the model, and we can access it as follows:

# %%
model.dataset

# %% [markdown]
# This can stack indefintely. On disk, the link to the dataset is stored as a symbolic link:

# %%
# !ls -lhaG --color=always {model.path} | sed -re 's/^[^ ]* //'

# %%
