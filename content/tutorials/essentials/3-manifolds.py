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
# # Manifolds

# %%
import latenta as la

# %% [markdown]
# So far, we have focused on models that basically look like this:

# %% [markdown]
# ![](_static/previous.svg)

# %% [markdown]
# But what if our cellular variables are also unknown, i.e. latent?

# %% [markdown]
# ![](_static/now.svg)

# %% [markdown]
# While many manifold models are relatively easy to implement, the main difficulty lies in the interpretability. Especially when different **cellular processes** are happening at the same time in a cell, a latent variable will try to explain all of them.
#
# What is therefore often required is the inclusion of prior knowledge that can help with disentangling different cellular processes

# %%
