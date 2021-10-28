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
# # The essentials of modelling with latenta

# %% [markdown]
# With a background in single-cell biology, chances are high you already created/worked with many types of models:
# - Clustering
# - Dimensionality reduction
# - Differential expression
# - Trajectories
# - RNA velocity

# %% [markdown]
# All these models have several aspects in common: we try to explain what we observe (the transcriptome of many cells), using things we do not know. These unknowns can be specific to a gene (such as a gene's fold change or dynamics during a trajectory), to a cell (such as a cells position in a reduced space or its pseudotime) or to both (e.g. the future state of a cell).
#
# While these models are powerful, they are all mostly one-dimensional in nature: we create a single clustering, trajectory or embedding that tries to explain one or two modalities (often the transcriptome). We easily hit roadblocks if we try to explain more than one or two aspects of the cell, both in terms of cellular processes (differentiation, cell cycle, patient subtypes, ...), modalities (transcriptome, proteome, chromatin accessibility, ...), and the functions that connect them (smooth functions, deconvolutions, interactions, ...). Such questions can be:
#
# * How does the remaining variation look like once I model («regress out») a time series, batch effect and cell cycle?
# * Does my genetic perturbation affect target gene expression in a linear, sigmoid, exponential or more complex manner?
# * How is the cell cycle distributed with zonation in a tissue?
# * Are there genes that are induced after 6 hours of treatment commonly expressed in all celltypes?
# * Does the surgery «interact» with the cell cycle?
# * Do my genetic perturbations interact with/influences cellular processes (cell cycle, differentiation...)? Is this in an additive, synergistic, or more complex _epistatic_ manner?
# * How many cells are cycling in a spatial spot, and in which phases are these cells?
# * Is my genetic perturbation inducing two or three cellular states?
# * ...
#
# To answer these, we need a more modular toolbox, which can model various cellular processes and states at the same time!

# %% [markdown]
# ## But why would I want to model data?

# %% [markdown]
# Why model data? You as a user probably have either of four goals in mind: description, discovery, understanding, or prediction:
#
# - For **description** and **discovery**, the complexity of the model depends on the question. Sometimes, a simpler model that normalizes and visualizes the data can suffice. This is the case if we want to describe and discover the cell types within a sample, as the actual descriptive and discovery work is then done by us humans based on the 2D visualization. There are plenty of tools in the single-cell field that can do this, and if you only want to describe data, latenta may be overkill.
# However, for more complex but (hopefully) useful descriptions, such as those encompassing multiple cellular processes and modalities, the complexity of the model increases. In those cases, a more exact description as created by latenta can be important because it's hard to visualize and interpret these effects by humans objectively.
# - For **prediction**, we mainly care about creating generalizable models. Generalizability means that we want a model that not only works on the cells from our dataset, but we also want to make good predictions about any future cells (and ideally, patients). When predicting, we typically do not care much about interpretability. However, just like we often want to know the mechanism of a potential drug, having an interpretable model can also be useful when making predictions.
# - For **understanding**, we care both about both interpretability and generalizability. For this reason, a model that is useful for understanding is often also good at making predictions.
#
# Interpretable models, however often:
#   - Require a bit more data because there are more free parameters
#   - Require more effort from the modeler. It's hard to create a "black-box" interpretable model as that would require an impossible amount of data. Instead, we specify some possible models, and by doing that, we bring in some prior knowledge and what interpretability means for us.
#
# Latenta is primarily made to create models enabling more complex descriptions and therefore helping to move towards understanding. It allows the (computational) biologist to create, infer, interpret, connect and share these models.

# %% [markdown]
# ```{note}
# Latenta is very related to and borrows elements from tools such as stan, TensorFlow, and torch, which are different packages for statistical modeling/machine learning. However, we focused on making the entry barrier to apply these tools to biological data much lower by focussing on:
# - Modularity, so that you can easily reuse parts made by others without having to understand them (although it can help!)
# - Dummy-proofing, so that it is obvious what is wrong with a model early on
# - Interpretability so that we can easily see what our model is doing
#
# We hope that these aspects make it easier to bridge machine learning and biology
# ```

# %% [markdown]
# Modeling can sometimes be frustrating if you lack the necessary background. Whatever you do and whichever tool you use, always keep in mind:
# - Modeling should be fun, and not having fun often means you lack some knowledge or that you are asking too much from the data
# - Experience is the source of knowledge
# - The best experience is the one you built up slowly but surely, starting from what others have done and then trying to adapt and experiment
#
# Latenta is made with these elements in mind. It's relatively straightforward to reuse other people's models without caring about the details. But you can then delve into parts of a model, change some elements where needed and combine different models. In the end, after some effort, knowledge and expertise will come!

# %% [markdown]
# ## Different layers of latenta
#
# Latenta has different layers of complexity:
#
# - At the highest level, latenta contains workflows and pipelines for certain common tasks. These workflows are not black boxes but are still completely customizable.
# - At the medium level, we provide ways to construct models for several common cellular processes and modalities, such as transcriptomics, proteomics, chromatin accessibility, trajectories, cell cycle, etc.
# - At a lower level, we provide ways to create individual variables and connect them with each other. We also provide various functions to help interpret the results.
# - At the lowest level, we provide ways to create custom sets of variables, posteriors, etc.
#
# If you're relatively new to modeling, it's often easier to start working with the highest level functions and slowly learn to work with lower (but more flexible) modeling techniques. Nonetheless, in these tutorials we will give an overview on each level starting from the bottom so that you're familiar with the underlying statistical and machine learning concepts.

# %% [markdown]
# Any model is a mathematical representation of reality, and building models does therefore require some mathematical intuition. Although latenta may make this easier, it would be unwise to abstract away every mathematical concept completely. Some probability, statistics, and machine learning concepts are too central for modeling (and, in fact, biology!) to hide. In these tutorials, we often refer you to relevant "explanation" resources (denoted by a see also sign) where you can improve your understanding of some of these concepts.

# %%
