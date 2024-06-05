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

adata = la.data.load_egr1()
adata.obs["log_overexpression"] = np.log1p(adata.obs["overexpression"])
adata.var["label"] = adata.var["symbol"]
adata.raw = adata

adata = adata[adata.obs["phase"] == "G1"]


# %% [markdown]
# All models are incorrect, but some are useful.
#
# But how do we know which model is the most useful? If we are interested in understanding, this is typically the model that is simple, while still explaining the data well: 

# %% [markdown]
# - It should be generalizable, i.e. we should still get the model if we would gather more data.
# - It should be relevant, i.e. it should change 
# - It should explain the data well, i.e. 

# %% [markdown]
# ## Creating the models

# %% [markdown]
# We will illustrate model selection with perhaps the simplest model possible: differential expression. We are given an outcome, the overexpression of *Egr1*, and now want to know which genes are differential. To do this, we will first create 2 models: a linear response and a constant response. And determine whether they follow the three main criteria.

# %% [markdown]
# At this stage, the following model-building sections should not hold any mysteries for you and are undocumented.

# %%
models = {}

# %%
overexpression = la.Fixed(adata.obs["log_overexpression"], label="overexpression")

# %% [markdown] tags=[]
# ### Model 1: Linear

# %%
transcriptome = lac.transcriptome.create_observation_from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Linear(
    overexpression, a=True, definition=foldchange.value_definition
)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam()
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot(show = True)

# %%
transcriptome_predictive = la.posterior.vector.VectorPredictive(transcriptome)
transcriptome_predictive.sample(5)

# %%
overexpression_predictive = la.posterior.scalar.ScalarPredictive(overexpression)
overexpression_predictive.sample(5)

# %%
overexpression_conditional = la.posterior.scalar.ScalarVectorConditional(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu,
    predictive=overexpression_predictive,
)
overexpression_conditional.sample(10)
overexpression_conditional.sample_random(10)
overexpression_conditional.predictive
overexpression_conditional.sample_empirical()

# %%
overexpression_conditional.plot_features()

# %% [markdown]
# Let's now add our results for a linear model in `models["linear"]`

# %%
models["linear"] = {
    "root": transcriptome,
    "overexpression_conditional": overexpression_conditional,
    "transcriptome_predictive": transcriptome_predictive,
}

# %% [markdown] tags=[]
# ### Model 2: Constant

# %% [markdown]
# We want to compare our models to a null hypothesis. Since our model we want to find the genes that are differential with *Myod1* overexpression (and how, linear, spline) our null hypothesis is that there is no change with overexpression, i.e the fold change is null.

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.create_observation_from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.scalar.Constant(
    overexpression, definition=foldchange.value_definition
)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam()
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot(show = True)

# %%
transcriptome_predictive = la.posterior.vector.VectorPredictive(transcriptome)
transcriptome_predictive.sample(20)

# %%
overexpression_predictive = la.posterior.scalar.ScalarPredictive(overexpression)
overexpression_predictive.sample(5)

# %%
transcriptome_predictive.samples.keys()

# %%
overexpression_conditional = la.posterior.scalar.ScalarVectorConditional(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu,
    predictive=overexpression_predictive,
)
overexpression_conditional.sample(10)
overexpression_conditional.sample_random(10)
overexpression_conditional.predictive
overexpression_conditional.sample_empirical()

# %%
overexpression_conditional.plot_features()

# %%
models["constant"] = {
    "root": transcriptome,
    "overexpression_conditional": overexpression_conditional,
    "transcriptome_predictive": transcriptome_predictive,
}

# %% [markdown]
# ## Generalizability: Bayes factor

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
# Of course, this is all balanced out with the presence of predictive data. If we have a lot of data, more complex models will be able to overcome this penality if they explain the data well. In other words:
#
# > Extraordinary claims require extraordinary evidence
#
# You may have noticed a pattern here: in the [regression tutorial](2-regression) we also talked about balancing the needs of the priors. This is no coincidence as both model inference and model comparison are highly linked mathematically. Indeed, to estimate Bayes factors we use exactly the same loss function as we typically use for inference: the evidence lower bound or ELBO. This is only an estimation of this measure, as calculating the exact Bayes factor is both mathematically impossible and computationally intractable even for moderately complex models.
#
# One final note: in the presence of very large data, almost every more complex model will be significant. You may have already seen this when doing differential expression in single-cell data, as a large number of genes are significant even though the differences between populations may be minor. This is not a problem of design per se (although some may disagree :citel:{squair_confronting_2021}). Rather, we also need to consider biological significance in our models, such as requiring at least a two fold difference.

# %%
import pandas as pd

# %% [markdown]
# ### Comparing constant versus all other models

# %% [markdown]
# First, we can find generally differentially expressed genes by simply comparing the constant model (null hypothesis) with all other models:

# %%
elbo_genes = pd.DataFrame(
    {
        model_id: model["transcriptome_predictive"].elbo_features.to_pandas()
        for model_id, model in models.items()
    }
)

# %%
bfs_constant_linear = elbo_genes["linear"] - elbo_genes["constant"]
scores_constant_linear = bfs_constant_linear.rename("bf").to_frame()

# %%
bfs_constant_linear.plot(kind="kde")

# %%
scores_constant_linear["symbol"] = adata.var["symbol"][scores_vsHo.index]
scores_constant_linear = scores_constant_linear.sort_values("bf", ascending=False)
scores_constant_linear

# %%
plot = models["linear"]["overexpression_conditional"].plot_features(
    feature_ids=scores_constant_linear.index[:5], show = True
)
plot.suptitle("Differential genes")
plot = models["linear"]["overexpression_conditional"].plot_features(
    feature_ids=scores_constant_linear.query("bf > log(10)").index[-5:], show = True
)
plot.suptitle("Barely differential genes")
plot = models["linear"]["overexpression_conditional"].plot_features(
    feature_ids=scores_constant_linear.index[-5:], show = True
)
plot.suptitle("Non-differential genes")

# %% [markdown]
# :::{note}
#
# Working with these different models and posteriors may feel quite cumbersome, with tons of repeated code for each model. We agree! That's why in the next tutorial, you're going to see our approach at making these things easier to scale up and maintaining persistence of models across session.
#
# :::

# %% [markdown]
# ## Relevance: investigating the effect size using the conditional posterior

# %% [markdown]
# While the BF ensures against overfitting on limited data, which is important when samples signal over noise is low such as in perturbatonal studies, it does have limitations. With enough data, a BF will tend to select the more complex model, a limitation it shares with p-values [@gelmanPhilosophyPracticeBayesian2013, matteiParsimoniousTourBayesian2020, navarroPersonalEssayBayes2020], and an issue also important for differential expression [@pullinComparisonMarkerGene2022]. Moreover, a BF simply tells that one or more models are better than other models at explaining the data, but it does not ensure that the chosen models do explain the data well. This is particularly important in situations where the data truly indicates something is happening, e.g. an interaction between cell cycle and TF dosage, but none of the available models approximate the true biology very well. In those cases, the BF may still select a “best” among all equally “bad” models.

# %% [markdown]
# This does not mean the BF is a bad measure, it simply means that the BF alone is not enough.

# %%
expression_constant = models["constant"]["overexpression_conditional"].mean_interpretable
expression_linear = models["linear"]["overexpression_conditional"].mean_interpretable

# %%
scores_constant_linear["fc"] = 2 ** (np.abs(np.log2(foldchange_linear) - np.log2(foldchange_constant)).max("cell").to_pandas())
scores_constant_linear["lfc"] = np.log2(scores_constant_linear["fc"])
scores_constant_linear["ac"] = np.abs(foldchange_linear - foldchange_constant).max("cell").to_pandas()
scores_constant_linear["lac"] = np.log2(scores_constant_linear["ac"])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.pairplot(
    vars = ["lfc", "lac", "bf"],
    data = scores_constant_linear
)

# %%
gene_ids = scores_constant_linear.query("((ac < 1) & (fc < 1.5))").sort_values("bf", ascending = False).index[:5]

plot = models["linear"]["overexpression_conditional"].plot_features(
    gene_ids, show = True, legend = False
)
plot.suptitle("Significant but irrelevant linear genes, linear model")

# %% [markdown]
# ## Predictability: model criticism 

# %% [markdown]
# We want to compare our models to a null hypothesis. Since our model we want to find the genes that are differential with *Myod1* overexpression (and how, linear, spline) our null hypothesis is that there is no change with overexpression, i.e the fold change is null.

# %% [markdown]
# ### Model 3: NN

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.create_observation_from_adata(adata)
foldchange = transcriptome.find("foldchange")

foldchange.overexpression = la.links.nn.feedforward.create_feedforward_bnn(
    overexpression, foldchange
)
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], la.infer.optim.Adam()
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot(show = True)

# %%
transcriptome_predictive = la.posterior.vector.VectorPredictive(transcriptome)
transcriptome_predictive.sample(100)

# %%
overexpression_predictive = la.posterior.scalar.ScalarPredictive(overexpression)
overexpression_predictive.sample(5)

# %%
transcriptome_predictive.samples.keys()

# %%
overexpression_conditional = la.posterior.scalar.ScalarVectorConditional(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu,
    predictive=overexpression_predictive,
)
overexpression_conditional.sample(10)
overexpression_conditional.sample_random(10)
overexpression_conditional.predictive
overexpression_conditional.sample_empirical()

# %%
overexpression_conditional.plot_features()

# %%
models["nn"] = {
    "root": transcriptome,
    "overexpression_conditional": overexpression_conditional,
    "transcriptome_predictive": transcriptome_predictive,
}

# %% [markdown]
# ### Calculating the information-gain rato

# %%
likelihood_genes = pd.DataFrame(
    {
        model_id: model["transcriptome_predictive"].likelihood_features.to_pandas()
        for model_id, model in models.items()
    }
)

# %%
scores_constant_linear["igr"] = (likelihood_genes["nn"] - likelihood_genes["constant"]) / (likelihood_genes["linear"] - likelihood_genes["constant"])
scores_constant_linear["ig"] = likelihood_genes["nn"] - likelihood_genes["constant"]
scores_constant_linear["ig2"] = likelihood_genes["linear"] - likelihood_genes["constant"]

# %%
scores_oi = scores_constant_linear.query("(bf > log(10)) & ((fc > 1.5) | (ac > 1.5))")

# %%
sns.scatterplot(scores_oi["ig"], scores_oi["ig2"], hue = scores_oi["igr"])

# %%
gene_ids = scores_constant_linear.query("(bf > log(10)) & ((fc > 1.5) | (ac > 1.5))").sort_values("igr", ascending = False).index[:10]
gene_ids = scores_constant_linear.query("(bf > log(10)) & ((fc > 1.5) | (ac > 1.5))").sort_values("bf", ascending = False).index[:10]

# %%
scores_constant_linear.query("(bf > log(10)) & ((fc > 1.5) | (ac > 1.5))").sort_values("igr", ascending = False)

# %%
models["nn"]["overexpression_conditional"].plot_features(gene_ids)

# %%
models["linear"]["overexpression_conditional"].plot_features(gene_ids)

# %% [markdown]
# ## Adding more models

# %% [markdown] tags=[]
# ### Model 4: Spline

# %%
overexpression = overexpression.reset().clone()

# %%
transcriptome = lac.transcriptome.create_observation_from_adata(adata)
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
transcriptome_predictive = la.posterior.vector.VectorPredictive(transcriptome)
transcriptome_predictive.sample(100)

# %%
overexpression_predictive = la.posterior.scalar.ScalarPredictive(overexpression)
overexpression_predictive.sample(5)

# %%
overexpression_conditional = la.posterior.scalar.ScalarVectorConditional(
    overexpression,
    transcriptome,
    interpretable=transcriptome.p.mu,
    predictive=overexpression_predictive,
)
overexpression_conditional.sample(10)
overexpression_conditional.sample_random(10)
overexpression_conditional.predictive
overexpression_conditional.sample_empirical()

# %%
overexpression_conditional.plot_features()

# %% [markdown]
# Let's now add our results for a linear model in `models["spline"]`

# %%
models["spline"] = {
    "root": transcriptome,
    "overexpression_conditional": overexpression_conditional,
    "transcriptome_predictive": transcriptome_predictive,
}

# %% [markdown]
# ### Comparing spline and linear

# %% [markdown]
# Bayes factor:

# %%
bfs_linear_spline = (elbo_genes["spline"] - elbo_genes["linear"]).rename("bf").to_frame()
scores_linear_spline = bfs_linear_spline.max(1).rename("bf").to_frame()

scores_linear_spline["bf_constant"] = bfs_linear_spline.max(1)
scores_linear_spline = scores_linear_spline.query("bf_constant > log(10)")

# %% [markdown]
# Effect size:

# %%
expression_constant = models["constant"]["overexpression_conditional"].mean_interpretable
expression_linear = models["linear"]["overexpression_conditional"].mean_interpretable

# %%
scores_linear_spline["fc"] = 2 ** (np.abs(np.log2(expression_linear) - np.log2(expression_constant)).max("cell").to_pandas())
scores_linear_spline["lfc"] = np.log2(scores_linear_spline["fc"])
scores_linear_spline["ac"] = np.abs(expression_linear - expression_constant).max("cell").to_pandas()
scores_linear_spline["lac"] = np.log2(scores_linear_spline["ac"])

# %% [markdown]
# IGR:

# %%
likelihood_genes = pd.DataFrame(
    {
        model_id: model["transcriptome_predictive"].likelihood_features.to_pandas()
        for model_id, model in models.items()
    }
)

# %%
scores_linear_spline["igr"] = (likelihood_genes["nn"] - likelihood_genes["constant"]) / (likelihood_genes["spline"] - likelihood_genes["constant"])

# %%
scores_linear_spline["symbol"] = adata.var["symbol"][scores_linear_spline.index]
scores_linear_spline = scores_linear_spline.sort_values("bf", ascending=False)
scores_linear_spline

# %%
nonlinear = scores_linear_spline.query("(bf > log(10)) & ((fc > 1.5) | (ac > 1.5))").sort_values("bf", ascending=False)
linear = scores_linear_spline.sort_values("bf", ascending=True)

# %%
scores_linear_spline["igr"][nonlinear.index], scores_constant_linear["igr"][nonlinear.index]

# %%
plot = models["spline"]["overexpression_conditional"].plot_features(
    feature_ids=nonlinear.index[:5], show = True
)
plot.suptitle("Non-linear genes, spline model")
plot = models["linear"]["overexpression_conditional"].plot_features(
    feature_ids=nonlinear.index[:5], show = True
)
plot.suptitle("Non-linear genes, linear model")

# %%
plot = models["spline"]["overexpression_conditional"].plot_features(
    feature_ids=linear.index[:5], show = True
)
plot.suptitle("Linear genes, spline model")
plot = models["linear"]["overexpression_conditional"].plot_features(
    feature_ids=linear.index[:5], show = True
)
plot.suptitle("Linear genes, linear model")

# %%
sns.scatterplot(scores_linear_spline["igr"], scores_constant_linear["igr"])

# %% [markdown]
# ### Relevance

# %%
