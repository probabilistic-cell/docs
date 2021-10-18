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
# # Transcriptome models

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac
import laflow as laf
import numpy as np

# %% [markdown]
# lacell not only contains classes that help with model creation, but also with workflow creation. For example, if we're working with transcriptomics data, we will often inherit from {class}`~lac.transcriptome.TranscriptomeDataset` and {class}`~laf.Model`. These already contain the core elements for storing and inferring transcriptomics models.

# %% [markdown]
# We'll will showcase this with the same dataset as from the [essentials tutorial](/tutorials/essentials/0-modelling.html).

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

# %%
import tempfile
import pathlib

project_root = pathlib.Path(tempfile.TemporaryDirectory().name)
project_root.mkdir()
laf.set_project_root(project_root)  # sets the default project root

# %%
dataset = lac.transcriptome.TranscriptomeDataset("dataset")

# %% [markdown]
# It contains information from one modality, namely the transcriptome:

# %%
dataset.transcriptome

# %%
dataset.transcriptome.from_adata(adata=adata)

# %%
dataset.transcriptome

# %%
dataset


# %%
class ConstantModel(laf.Model):
    default_name = "constant"
    dataset = laf.FlowObj()

    # because we are overwriting the create from the parent classes
    # we do not need to specify @laf.Step here
    def create(self, output, X, obs, var):
        output = super().create_(output, X, obs, var)

        transcriptome = output.model_initial

        # define the model as before
        overexpression = la.Fixed(obs["log_overexpression"], label="overexpression")
        foldchange = transcriptome.find("foldchange")

        foldchange.overexpression = la.links.scalar.Constant(
            overexpression, definition=foldchange.value_definition
        )
        return output

    model = laf.Model.model

    overexpression_observed = laf.LatentaObj(db={model})
    overexpression_causal = laf.LatentaObj(db={model})

    @laf.Step(
        laf.Inputs(model), laf.Outputs(overexpression_observed, overexpression_causal)
    )
    def interpret_overexpression(self, output, model):
        overexpression = model.find("overexpression")

        overexpression_observed = la.posterior.scalar.ScalarObserved(overexpression)
        overexpression_observed.sample(5)
        output.overexpression_observed = overexpression_observed

        overexpression_causal = la.posterior.scalar.ScalarVectorCausal(
            overexpression,
            model,
            interpretable=model.p.mu.expression,
            observed=overexpression_observed,
        )
        overexpression_causal.sample(10)
        overexpression_causal.sample_random(10)
        overexpression_causal.sample_empirical()
        output.overexpression_causal = overexpression_causal

        return output


class LinearModel(ConstantModel):
    # we change the default name, as to make sure this model is put in a different folder
    default_name = "linear"

    def create(self, output, X, obs, var):
        # we can access the inherited function by adding a "_" at the end
        output = super().create_(output, X, obs, var)

        # extract the model_initial from the output
        model_initial = output.model_initial

        # now we can further adapt the model to our wish
        foldchange = model_initial.find("foldchange")
        overexpression = model_initial.find("overexpression")

        foldchange.overexpression = la.links.scalar.Linear(
            overexpression, a=True, definition=foldchange.value_definition
        )

        # again return the output
        # because we only adapted the model inplace, we do not need to update the output
        return output


class SplineModel(ConstantModel):
    default_name = "spline"

    def create(self, output, X, obs, var):
        output = super().create_(output, X, obs, var)

        model_initial = output.model_initial

        foldchange = model_initial.find("foldchange")
        overexpression = model_initial.find("overexpression")

        foldchange.overexpression = la.links.scalar.Spline(
            overexpression, definition=foldchange.value_definition
        )

        return output


# %%
model = LinearModel(dataset=dataset)
model.create()
model.infer()
model.interpret_transcriptome()

# %%
model.interpret_overexpression()

# %%
model

# %%

# %%

# %%
