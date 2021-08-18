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
# <img src="_static/regression.svg">|

# %% [markdown]
# But what if our cellular variables are also unknown:

# %% language="html"
# <canvas id="canvas", width = 100, height = 200>
#
# </canvas>

# %% language="javascript"
# const rc = rough.canvas(document.getElementById('canvas'));
# rc.rectangle(10, 10, 200, 200); // x, y, width, height

# %% [markdown]
# So far, we have 
