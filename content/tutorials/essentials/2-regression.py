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
# # Regression

# %%
import latenta as la
import scanpy as sc
import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables)...

# %%
adata = la.data.load_myod1()
adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])

# %% [markdown]

# ## Easy things first: Linear regression

# %% [markdown]
# ### Creating models
# In the last tutorial, we learned the main types of variables, and how to connect these to construct a model:
# - You either specify them at initialization: `la.Observation(p = ...)`
# - Or connect them afterwards: `observation.p = ...`
#
# We can now use these concepts to build a basic model of a transcriptome:

# %%
genes = la.Dim(adata.var.index)
cells = la.Dim(adata.obs.index)

# %%
overexpression = la.Fixed(adata.obs["log_overexpression"], label = "overexpression", symbol = "overexpression")

# %%
expression = la.links.scalar.Linear(
    overexpression,
    a = True,
    b = True,
    label = "expression",
    definition = la.Definition([cells, genes]),
    transforms = [la.transforms.Exp()]
)

# %% [markdown]
# Although the latent slope and baseline were created automatically for us by specifying `a = True, b = True`, it's important to remember that we could have easily created these variables ourselves:
#
# ```python
# slope = la.Latent(
#     p = la.distributions.Normal(scale = la.Parameter(1., transforms = [la.transforms.Exp()])),
#     definition = la.Definition([genes])
# )
# ```

# %%
dispersion = la.Latent(
    la.distributions.LogNormal(
        la.Parameter(0.),
        la.Parameter(1., transforms = [la.transforms.Exp()])
    ),
    definition = la.Definition([genes])
)

# %%
transcriptome_p = la.distributions.NegativeBinomial2(
    mu = expression,
    dispersion = dispersion
)

# %%
transcriptome = la.Observation(adata.X, transcriptome_p, definition = la.Definition([cells, genes]), label = "transcriptome")

# %%
transcriptome.plot()

# %% [markdown]
# Note the many parameters that form the leaves of our model. But, why are there so many even for the simples of a simple case of linear regression?
#
# Let's remind ourselves what we're actually trying to accomplish. We're trying to create a good model of our observations. There are however many models that will provide a very good fit of the data equally. For example, we could just give the actual count matrix as input to the negative binomial and this trivial model would fit extremely well.
#
# So we don't just want a model. We want a model that can explain our observation well, while being both generalizeable and interpretable. And to accomplish his, we have to limit the flexibility that our model can have. You have already done this in two ways:
#
# - Hard priors are those that completely constrain the model. For example, by specifying a linear function we don't allow any non-linearities
# - Soft priors are those that simply push the latent variables towards more likely values. For example, we want to discourage extreme slopes as these are likely caused by lack of data.
#
# All these parameters thus serve two purposes: the parameters of the variational distributions $q$ will try to explain the observations while also remaining faithful to the prior distributions $p$. The parameters of $p$ on the other hand will try to accomodate the parameters of $q$ as well as possible, but it cannot do this perfectly as these parameters are shared across all genes. It's this pushing and pulling between priors and variational distributions that prevent overfitting and underfitting of the model. At the same time, we get some estimates of the uncertainty of our latent variables for free!

# %% [markdown]
# But how do we infer these parameters in practice? We have to find a solution that best balances the needs of the prior distribution with those of the observations. And one of the fastest ways to do that is to use gradient descent, which starts from an initial value and then tries to move these initial values slowly but surely into better values. These tasks are fullfilled by a loss function (`ELBO`), an optimizer (`Adam`), and an overarching training class (`SVI`):

# %%
inference = la.infer.svi.SVI(
    transcriptome,
    [la.infer.loss.ELBO()],
    la.infer.optim.Adam(lr = 0.05)
)
trainer = la.infer.trainer.Trainer(inference)

# %% [markdown]
# We can now train for a couple of iterations or until the loss function has reached a plateau (= converged):

# %%
trace = trainer.train(10000)
trace.plot();

# %% [markdown]
# You can check that our values have changed:

# %%
transcriptome.p.mu.a.q.loc.run()
transcriptome.p.mu.a.q.loc.value_pd.head()

# %% [markdown]
# ### Observed posteriors

# %%

# %% [markdown]
# ### Causal posteriors

# %% [markdown]
# ### Perturbed posteriors

# %% [markdown]
# ## Using _lacell_ to make model creation easier

# %%

# %% [markdown]
# ## More complex regression problems


# %% [markdown]
# ### Multiple inputs

# %% [markdown]
# ### Non-linear


