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
# # The essentials of modelling with latenta

# %% [markdown]
# With a background in single-cell biology, chances are high you already created many types of models:
# - Clustering
# - Dimensionality reduction
# - Differential expression
# - Trajectories
# - RNA velocity

# %% [markdown]
# All these models have several aspects in common: we try to explain what we observe (the transcriptome of many cells), using things we do not know. These unknowns can be specific to a gene (such as a gene's fold change or dynamics during a trajectory), a cell (such as a cells position in a reduced space or its pseudotime) or both (e.g. the future state of a cell).

# %% [markdown]
# Why model data? You as a user probably have either of two goals in mind: prediction or understanding.
#
# - For **prediction**, we mainly care about make generalizable models. Generalizability means we want a model that not only works on the cells we just killed in an experiment, but also want to make good predictions about any future cells (and ideally, patients). When predicting, we typically do not care much about interpretability. However, just like we often want to know the mechanism of a potential drug, having an interpretable model can also be useful when making predictions.
# - For **understanding**, we care both about both interpretability and generalizability. For this reason, a model that is useful for understanding is often also good at making predictions.
#
# Interpretable models however often:
#   - Require a bit more data, because there are more free parameters
#   - Require more effort from the modeller. It's hard to create a "black-box" interpretable model as that would require an impossible amount of data. Rather, we specify some possible models and by doing that we bring in some prior knowledge and what interpretability means for us.
#
# Latenta is primarily made to create models for understanding: to make it easier for the (computational) biologist to create, infer, interpret and share models.

# %% [markdown]
# ```{note}
# Latenta is very related, and borrows elements, from tools such as stan, tensorflow and torch. However, we focused on making the entry barrier much lower, by focussing on:
# - Modularity, so that you can easily reuse parts made by others without having to understand them (although it can help!)
# - Dummy-proofing, so that it is obvious what is wrong with a model early on
# - Interpretability, so that we can easily see what our model is doing
#
# Our hope is that these aspects make it easier to bridge machine learning and biology
# ```

# %% [markdown]
# Modelling can sometimes be frustrating if you lack the necessary background. Whatever you do, and whichever tool you use, always keep in mind:
# - Modelling should be fun, and not having fun often means you lack some knowledge or that you are asking too much from the data
# - Experience is the source of knowledge
# - The best experience is the one you built up slowly but surely, starting from what others have done and then try to adapt and experiment
#
# Latenta is made with these elements in mind. It's relatively straightforward to reuse other people's models without caring about the details. But you can then delve into parts of a model, change some elements where needed and combine different models. In the end, after some effort, knowledge will come.

# %% [markdown]
# Any model is a mathematical representation of reality, and building models does require some mathematics. Although latenta may make this easier, it would be unwise to completely abstract away every mathematical concept. Some probability, statistics and machine learning concepts are simply too central for modelling (and, in fact, biology!) to hide. In these tutorials, we often refer you to relevant "explanation" resources (denoted by a see also sign) where you can improve your understanding of some of these concepts.

# %% [markdown]
# Modelling typically consists of several steps:
# 1. Creating known and unknown variables
# 2. Connect these variables, i.e. creating the model
# 3. Infer the value of the unknown variables
# 4. Interpret the model
#
# In this tutorial, we give an overview of each step. More details on using latenta to interpret specific types of datasets can be found in the [tutorials](/tutorials), while detailed explanations on specific problems (e.g. the cell cycle) can be found in the [user guide](/guide).

# %%
