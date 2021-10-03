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
# # Model selection

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac
import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables).

# %%
import scanpy as sc

adata = la.data.load_myod1()
adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])
adata.var["label"] = adata.var["symbol"]
adata.raw = adata


# %% [markdown]
# When we say that a model is statistically significant, what we actually mean is that one model is much more likely to be true than another _simpler_ model. In classical statistical testing these two models are called the alternative hypothesis and the null hypothesis.
#
# While statistical testing is powerful for some models, it often falls short when we try to do anything more complex. If we for example include any cellular latent variables, it becomes impossible to calculate any reasonable p-value. Furthermore, if we want to compare many different models.
#
# One main reason to thus use probabilistic modelling is that it allows us to generalize statistical testing to a technique called model selection. In this way, we can even In a probabilistic setting, the generalization of statistical testing is called model selection.

# %% [markdown]
# ## Creating the models

# %% [markdown]
# We will illustrate model selection with perhaps the simplest model possible: differential expression. We are given an outcome, the overexpression of Myod1, and now want to know which genes are differential. To do this, we will create 3 models: a linear response, a non-linear spline response, and a constant response.

# %% [markdown]
# At this stage, the following model-building sections should not hold any mysteries for you and are undocumented. Feel free to try out any extra models, e.g. a {class}`latenta.links.scalar.Switch`, or play with some of the parameters of the spline function to see how this affects the outcome.

# %%
models = {}

# %%
overexpression = la.Fixed(adata.obs["log_overexpression"], label="overexpression")

# %% [markdown] tags=[]
# ### Model 1: Linear

# %%
transcriptome = lac.transcriptome.Transcriptome.from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Linear(overexpression, a = True, definition=foldchange.value_definition)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
transcriptome_observed = la.posterior.vector.VectorObserved(transcriptome)
transcriptome_observed.sample(5)

# %%
overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
overexpression_observed.sample(5)

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression, transcriptome, interpretable = transcriptome.p.mu.expression, observed = overexpression_observed
)
overexpression_causal.sample(10)
overexpression_causal.sample_random(10)
overexpression_causal.observed
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features();

# %%
models["linear"] = {
    "root":transcriptome,
    "overexpression_causal":overexpression_causal,
    "transcriptome_observed":transcriptome_observed
}

# %% [markdown] tags=[]
# ### Model 2: Spline

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.Transcriptome.from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Spline(overexpression, definition=foldchange.value_definition)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
transcriptome_observed = la.posterior.vector.VectorObserved(transcriptome)
transcriptome_observed.sample(5)

# %%
overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
overexpression_observed.sample(5)

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression, transcriptome, interpretable = transcriptome.p.mu.expression, observed = overexpression_observed
)
overexpression_causal.sample(10)
overexpression_causal.sample_random(10)
overexpression_causal.observed
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features();

# %%
models["spline"] = {
    "root":transcriptome,
    "overexpression_causal":overexpression_causal,
    "transcriptome_observed":transcriptome_observed
}

# %% [markdown] tags=[]
# ### Model 3: Constant

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.Transcriptome.from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Constant(overexpression,definition=foldchange.value_definition)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot();

# %%
transcriptome_observed = la.posterior.vector.VectorObserved(transcriptome)
transcriptome_observed.sample(5)

# %%
overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
overexpression_observed.sample(5)

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression, transcriptome, interpretable = transcriptome.p.mu.expression, observed = overexpression_observed
)
overexpression_causal.sample(10)
overexpression_causal.sample_random(10)
overexpression_causal.observed
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features();

# %%
models["constant"] = {
    "root":transcriptome,
    "overexpression_causal":overexpression_causal,
    "transcriptome_observed":transcriptome_observed
}

# %% [markdown]
# ## Model selection

# %%
import pandas as pd

# %%
model_ids = list(models.keys())

# %%
elbo_genes = pd.DataFrame({model_id:model["transcriptome_observed"].elbo_features.to_pandas() for model_id, model in models.items()})
likelihood_genes = pd.DataFrame({model_id:model["transcriptome_observed"].elbo_features.to_pandas() for model_id, model in models.items()})

# %%
elbo_genes

# %% [markdown]
# ### Comparing constant versus all other models

# %% [markdown]
# First, we can find generally differentially expressed genes by simply comparing the constant model with all other models:

# %%
bfs = (elbo_genes.drop(columns = "constant") - elbo_genes["constant"].values[:, None])
scores = bfs.max(1).rename("bf").to_frame()

# %%
bfs.plot(kind = "kde")

# %%
scores["symbol"] = adata.var["symbol"][scores.index]
scores = scores.sort_values("bf", ascending = False)
scores

# %%
plot = models["spline"]["overexpression_causal"].plot_features(feature_ids = scores.index[:5]);
plot.suptitle("Differential genes")
plot = models["spline"]["overexpression_causal"].plot_features(feature_ids = scores.query("bf > log(10)").index[-5:]);
plot.suptitle("Barely differential genes")
plot = models["spline"]["overexpression_causal"].plot_features(feature_ids = scores.index[-5:]);
plot.suptitle("Non-differential genes")

# %% [markdown]
# :::{note}
#
# Working with these different models and posteriors may feel quite cumbersome, with tons of repeated code for each model. We agree! That's why in the next tutorial, you're going to see our approach at making these things easier to scale up and maintaining persistence of models across session.
#
# :::

# %% [markdown]
# ### Comparing spline and linear

# %%
scores = (elbo_genes["spline"] - elbo_genes["linear"]).rename("bf").to_frame()
scores["bf_constant"] = bfs.max(1)
scores = scores.query("bf_constant > log(10)")

# %%
scores["symbol"] = adata.var["symbol"][scores.index]
scores = scores.sort_values("bf", ascending = False)
scores

# %%
nonlinear = scores.query("(bf > log(10))").sort_values("bf_constant", ascending = False)
unclear = scores.query("(bf > log(10)) & (bf < log(30))").sort_values("bf_constant", ascending = False)
linear = scores.query("(bf < log(10))").sort_values("bf_constant", ascending = False)

# %%
plot = models["spline"]["overexpression_causal"].plot_features(feature_ids = nonlinear.index[:5]);
plot.suptitle("Non-linear genes")
plot = models["linear"]["overexpression_causal"].plot_features(feature_ids = nonlinear.index[:5]);
plot.suptitle("Non-linear genes")

plot = models["spline"]["overexpression_causal"].plot_features(feature_ids = unclear.index[:5]);
plot.suptitle("Unclear genes")
plot = models["linear"]["overexpression_causal"].plot_features(feature_ids = unclear.index[:5]);
plot.suptitle("Unclear genes")


plot = models["spline"]["overexpression_causal"].plot_features(feature_ids = linear.index[:5]);
plot.suptitle("Linear genes")
plot = models["linear"]["overexpression_causal"].plot_features(feature_ids = linear.index[:5]);
plot.suptitle("Linear genes")

# %%

# %%
