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
# When we say that a model is statistically significant, what we actually mean is that one model is much more likely to be true than another _simpler_ model. In classical statistical testing these two models are called the alternative and the null hypothesis.
#
# While statistical testing is powerful for some models, it often falls short when we try to do anything more complex. If we for example include any cellular latent variables, it becomes impossible to calculate any reasonable p-value.
#
# One main reason to thus use probabilistic modelling is that it allows us to generalize statistical testing to a technique called model selection. 

# %% [markdown]
# ## Creating the models

# %% [markdown]
# We will illustrate model selection with perhaps the simplest model possible: differential expression. We are given an outcome, the overexpression of *Myod1*, and now want to know which genes are differential. To do this, we will create 3 models: a linear response, a non-linear spline response, and a constant response.

# %% [markdown]
# At this stage, the following model-building sections should not hold any mysteries for you and are undocumented. Feel free to try out any extra models, e.g. a {class}`latenta.links.scalar.Switch`, or play with some of the parameters of the spline function to see how this affects the outcome.

# %%
models = {}

# %%
overexpression = la.Fixed(adata.obs["log_overexpression"], label="overexpression")

# %% [markdown] tags=[]
# ### Model 1: Linear

# %%
transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Linear(
    overexpression, a=True, definition=foldchange.value_definition
)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot(show = True)

# %%
transcriptome_observed = la.posterior.vector.VectorObserved(transcriptome)
transcriptome_observed.sample(5)

# %%
overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
overexpression_observed.sample(5)

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu.expression,
    observed=overexpression_observed,
)
overexpression_causal.sample(10)
overexpression_causal.sample_random(10)
overexpression_causal.observed
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features()

# %% [markdown]
# Let's now add our results for a linear model in `models["linear"]`

# %%
models["linear"] = {
    "root": transcriptome,
    "overexpression_causal": overexpression_causal,
    "transcriptome_observed": transcriptome_observed,
}

# %% [markdown] tags=[]
# ### Model 2: Spline

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Spline(
    overexpression, definition=foldchange.value_definition
)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot(show = True)

# %%
transcriptome_observed = la.posterior.vector.VectorObserved(transcriptome)
transcriptome_observed.sample(5)

# %%
overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
overexpression_observed.sample(5)

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu.expression,
    observed=overexpression_observed,
)
overexpression_causal.sample(10)
overexpression_causal.sample_random(10)
overexpression_causal.observed
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features()

# %% [markdown]
# Let's now add our results for a linear model in `models["spline"]`

# %%
models["spline"] = {
    "root": transcriptome,
    "overexpression_causal": overexpression_causal,
    "transcriptome_observed": transcriptome_observed,
}

# %% [markdown] tags=[]
# ### Model 3: Constant

# %% [markdown]
# We want to compare our models to a null hypothesis. Since our model we want to find the genes that are differential with *Myod1* overexpression (and how, linear, spline) our null hypothesis is that there is no change with overexpression, i.e the fold change is null.

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.TranscriptomeObservation.from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Constant(
    overexpression, definition=foldchange.value_definition
)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=5e-2)
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot(show = True)

# %%
transcriptome_observed = la.posterior.vector.VectorObserved(transcriptome)
transcriptome_observed.sample(5)

# %%
overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
overexpression_observed.sample(5)

# %%
overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu.expression,
    observed=overexpression_observed,
)
overexpression_causal.sample(10)
overexpression_causal.sample_random(10)
overexpression_causal.observed
overexpression_causal.sample_empirical()

# %%
overexpression_causal.plot_features()

# %% [markdown]
# We can now add our null hypothesis in `models["constant"]`

# %%
models["constant"] = {
    "root": transcriptome,
    "overexpression_causal": overexpression_causal,
    "transcriptome_observed": transcriptome_observed,
}

# %% [markdown]
# ## Model selection

# %% [markdown]
# ### The Bayes factor

# %% [markdown]
# How do we compare models? Because we are creating probabilistic models, it makes sense to also compare models using probabilities. But how to define such a measure?
#
# Let's again remind ourselves what we're trying to do. All we have is two things:
# - The data, which we further denote as $\mathcal{D}$. This contains both the values that we observe, but also the values that we inferred for the latent variables
# - One or more models, which we further denote as $\mathcal{M_i}$. This contains both the priors and structure ("hard" priors) of the model.
#
# To calculate the quality of a model $\mathcal{M_i}$, we have to calculate a probability of how well it explains the data $\mathcal{D}$ given the model, which we call the model evidence (or marginal likelihood):
#
# $$
# \mathit{evidence} = P(\mathcal{D} | \mathcal{M_i})
# $$
#
# The values for the evidence only become meaningful once we start to compare them between models. We often do this by calculating a ratio of evidences between two competing models, also called the Bayes factor $BF$:
#
# $$
# \mathit{BF} = \frac{P(\mathcal{D} | \mathcal{M_i})}{P(\mathcal{D} | \mathcal{M_j})}
# $$
#
# Bayes factors are quite intuitive: they tell you how much more likely one model is over the other given the data. We often use some rule-of-thumbs to set our cutoffs, e.g. {citel}`kass_bayes_1995`:

# %% [markdown]
#
#
# |    $\textit{BF}$   	|        Strength of evidence        	| $\log_{10}$$BF$ 	|   $\log BF$  	|
# |:---------:	|:----------------------------------:	|:------------:	|:------------:	|
# |  1 to 3.2 	| Not worth more than a bare mention 	|   0 to 1/2   	|   0 to ~1.1  	|
# | 3.2 to 10 	|             Substantial            	|   1/2 to 1   	| ~1.1 to ~2.3 	|
# | 10 to 100 	|               Strong               	|    1 to 2    	| ~2.3 to ~4.6 	|
# |   > 100   	|              Decisive              	|      > 2     	|    > ~4.6    	|

# %% [markdown]
# Although the goal of both p-values and Bayes factors may look similar, they do have a different interpretation. p-values quantify the evidence that we have to reject a null hypothesis, but does not provide any evidence in favor of an alternative hypothesis. In contrast, Bayes factors directly compare two models against each other, and include both the "null hypothesis model" and "alternative model" into the calculation.
#
# Furthermore, a Bayes factor considers the complete model _including prior distributions_. The further away the latent variable strays from the priors, the more it will be penalized. Moreover, the more latent variables we have, the more a model will be penalized as each of these latent variables has a prior distribution that needs to be satisfied. In other words, the more complex a model is, the higher the penality will be.
#
# Of course, this is all balanced out with the presence of observed data. If we have a lot of data, more complex models will be able to overcome this penality if they explain the data well. In other words:
#
# > Extraordinary claims require extraordinary evidence
#
# You may have noticed a pattern here: in the [regression tutorial](2-regression) we also talked about balancing the needs of the priors. This is no coincidence as both model inference and model comparison are highly linked mathematically. Indeed, to estimate Bayes factors we use exactly the same loss function as we typically use for inference: the evidence lower bound or ELBO. This is only an estimation of this measure, as calculating the exact Bayes factor is both mathematically impossible and computationally intractable even for moderately complex models.
#
# One final note: in the presence of very large data, almost every more complex model will be significant. You may have already seen this when doing differential expression in single-cell data, as a large number of genes are significant even though the differences between populations may be minor. This is not a problem of design per se (although some may disagree :citel:{squair_confronting_2021}). Rather, we also need to consider biological significance in our models, such as requiring at least a two fold difference.

# %%
import pandas as pd

# %%
model_ids = list(models.keys())

# %%
elbo_genes = pd.DataFrame(
    {
        model_id: model["transcriptome_observed"].elbo_features.to_pandas()
        for model_id, model in models.items()
    }
)
likelihood_genes = pd.DataFrame(
    {
        model_id: model["transcriptome_observed"].elbo_features.to_pandas()
        for model_id, model in models.items()
    }
)

# %%
elbo_genes

# %% [markdown]
# ### Comparing constant versus all other models

# %% [markdown]
# First, we can find generally differentially expressed genes by simply comparing the constant model (null hypothesis) with all other models:

# %%
bfs_vsHo = elbo_genes.drop(columns="constant") - elbo_genes["constant"].values[:, None]
scores_vsHo = bfs_vsCst.max(1).rename("bf").to_frame()

# %%
bfs_vsHo.plot(kind="kde")

# %%
scores_vsHo["symbol"] = adata.var["symbol"][scores_vsHo.index]
scores_vsHo = scores_vsHo.sort_values("bf", ascending=False)
scores_vsHo

# %%
plot = models["spline"]["overexpression_causal"].plot_features(
    feature_ids=scores_vsHo.index[:5], show = True
)
plot.suptitle("Differential genes")
plot = models["spline"]["overexpression_causal"].plot_features(
    feature_ids=scores.query("bf > log(10)").index[-5:], show = True
)
plot.suptitle("Barely differential genes")
plot = models["spline"]["overexpression_causal"].plot_features(
    feature_ids=scores.index[-5:], show = True
)
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
scores = scores.sort_values("bf", ascending=False)
scores

# %%
nonlinear = scores.query("(bf > log(10))").sort_values("bf_constant", ascending=False)
unclear = scores.query("(bf > log(10)) & (bf < log(30))").sort_values(
    "bf_constant", ascending=False
)
linear = scores.query("(bf < log(10))").sort_values("bf_constant", ascending=False)

# %%
plot = models["spline"]["overexpression_causal"].plot_features(
    feature_ids=nonlinear.index[:5], show = True
)
plot.suptitle("Non-linear genes")
plot = models["linear"]["overexpression_causal"].plot_features(
    feature_ids=nonlinear.index[:5], show = True
)
plot.suptitle("Non-linear genes")

# %%
plot = models["spline"]["overexpression_causal"].plot_features(
    feature_ids=linear.index[:5], show = True
)
plot.suptitle("Linear genes")
plot = models["linear"]["overexpression_causal"].plot_features(
    feature_ids=linear.index[:5], show = True
)
plot.suptitle("Linear genes")

# %%
plot = models["spline"]["overexpression_causal"].plot_features(
    feature_ids=unclear.index[:5], show = True
)
plot.suptitle("Unclear genes")
plot = models["linear"]["overexpression_causal"].plot_features(
    feature_ids=unclear.index[:5], show = True
)
plot.suptitle("Unclear genes")
