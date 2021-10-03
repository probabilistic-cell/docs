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
# # Workflows

# %% [markdown]
# For now we mainly looked at models in isolation. However, the true power of probabilistic modelling lies in its ability to move beyond the "one-model-fits-all" idea, but rather create a variety of models and connect or compare them in various ways. Indeed, even for a normal use cases, we already typically go up to a dozen models or more. In more complex experimental designs, which are becoming every more common, this can easily go up to a hundred or more.

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac
import numpy as np

# %% [markdown]
# Even in relatively simple use cases, managing different models, storing them across sessions, and let them depend on each other can be a pain. To make this easier, we created a small workflow manager in the `laflow` package. It borrows elements from other workflow managers such as snakemake or nextflow. However, it is much more focused on interactivity, modularization and model inheritance that, all concepts very central to the modelling we're doing here.
#
# The core tasks of laflow is:
# - **Persistence**, to store sure models and its derivatives on disk
# - **Model inheritance**, creation of similar models that only differ in certain aspects
# - **Connecting data**, to explicitely define what a model needs and where it gets this

# %%
import laflow as laf

# %% [markdown]
# We'll use the same dataset as [before](./1-variables).

# %%
import scanpy as sc

adata = la.data.load_myod1()
adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])
adata.var["label"] = adata.var["symbol"]
adata.raw = adata


# %% [markdown]
# ## Persistence

# %% [markdown]
# ::::{margin}
# :::{note}
# In the tutorials, we will always work from a temporary directory. In a real situation, the root directory will be typically your project's root directory or one of its subdirectories.
# :::
# ::::

# %%
import tempfile
import pathlib
project_root = pathlib.Path(tempfile.TemporaryDirectory().name)
project_root.mkdir()
laf.set_project_root(project_root) # sets the default project root

# %% [markdown]
# The basis of laflow is the {class}`laflow.Flow` class. Each instance we create is associated with a particular folder on disk, starting from a given root directory.

# %%
dataset = laf.Flow("dataset")
dataset.path

# %% [markdown]
# This folder will store information on each object, links to other objects or flows, and the objects themselves if they are labeled as persistent. We can easily store an object inside a flow by simply assigning it:

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
# ## Modularization

# %% [markdown]
# To make our workflows more modular, we use Python's OOP system to systematically let certain flows depend on each other and share common code. For example, a dataset in our example would be defined as follows:

# %%
class Dataset(laf.Flow):
    default_name = "dataset"
    adata = laf.PyObj()


# %% [markdown]
# We can initialize this dataset as before. Note that we do not provide a name in this case as we want to use the default "dataset" name:

# %%
dataset = Dataset()

# %%
dataset


# %% [markdown]
# We can now define our model as another Flow. It contains a link to the dataset flow, along with three methods that create, infer and interpret the model: 

# %%
class ConstantModel(laf.Flow):
    default_name = "constant"
    
    dataset = laf.FlowObj()
    
    root_initial = laf.LatentaObj(persistent = False)
    
    root = laf.LatentaObj()
    
    def create_root_initial(self, adata):
        overexpression = la.Fixed(adata.obs["log_overexpression"], label="overexpression")
        transcriptome = lac.transcriptome.Transcriptome.from_adata(dataset.adata)
        foldchange = transcriptome.find("foldchange")

        foldchange.overexpression = la.links.scalar.Constant(overexpression, definition=foldchange.value_definition)
        
        return transcriptome, 
        
    def infer_model(self, root_initial):
        root = root_initial.clone()
        with root.switch(la.config.device):
            inference = la.infer.svi.SVI(
                root, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
            )
            trainer = la.infer.trainer.Trainer(inference)
            trace = trainer.train(10000)
            
        return root, 
    
    transcriptome_observed = laf.LatentaObj()
    overexpression_observed = laf.LatentaObj()
    overexpression_causal = laf.LatentaObj()
            
    def interpret(self, root):
        transcriptome_observed = la.posterior.vector.VectorObserved(root)
        transcriptome_observed.sample(5)
        
        overexpression = root.find("overexpression")
        
        overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
        overexpression_observed.sample(5)
        
        overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
            overexpression, root, interpretable = root.p.mu.expression, observed = overexpression_observed
        )
        overexpression_causal.sample(10)
        overexpression_causal.sample_random(10)
        overexpression_causal.observed
        overexpression_causal.sample_empirical()
        
        return transcriptome_observed, overexpression_observed, overexpression_causal


# %%
model = ConstantModel()

# %%
model

# %%
model.dataset = dataset

# %%
model

# %%
model.dataset

# %%
model.root_initial,  = model.create_root_initial(model.dataset.adata)

# %%
model

# %%
model.root, = model.infer_model(model.root_initial)

# %%
model.transcriptome_observed, model.overexpression_observed, model.overexpression_causal = model.interpret(model.root)

# %%
model

# %%
model.overexpression_causal.plot_features();


# %% [markdown]
# We can now define our spline and linear models through inheritance, and only change the bits that are different. In this case, this is at the level of model creation:

# %%
class LinearModel(ConstantModel):
    default_name = "linear"
    
    def create_root_initial(self, adata):
        root_initial,  = super().create_root_initial(adata)
        
        foldchange = root_initial.find("foldchange")
        overexpression = root_initial.find("overexpression")
        
        foldchange.overexpression = la.links.scalar.Linear(overexpression, a = True, definition=foldchange.value_definition)
        
        return root_initial, 
    
class SplineModel(ConstantModel):
    default_name = "spline"
    
    def create_root_initial(self, adata):
        root_initial,  = super().create_root_initial(adata)
        
        foldchange = root_initial.find("foldchange")
        overexpression = root_initial.find("overexpression")
        
        foldchange.overexpression = la.links.scalar.Spline(overexpression, definition=foldchange.value_definition)
        
        return root_initial, 


# %%
model = LinearModel(dataset = dataset)
model.root_initial,  = model.create_root_initial(model.dataset.adata)
model.root, = model.infer_model(model.root_initial)
model.transcriptome_observed, model.overexpression_observed, model.overexpression_causal = model.interpret(model.root)

# %%
model = SplineModel(dataset = dataset)
model.root_initial,  = model.create_root_initial(model.dataset.adata)
model.root, = model.infer_model(model.root_initial)
model.transcriptome_observed, model.overexpression_observed, model.overexpression_causal = model.interpret(model.root)

# %%
model

# %% [markdown]
# ## Building pipelines

# %%
