# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Processes

# %% [markdown]
# The core of idea of latenta is modularization: there are many ways to model single-cell data, but instead of creating several tools and methods for each model, we create a modular interface that can combine and connect the different types of models. While

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac
import laflow as laf
import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables).

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
cellcycle_genes = lac.transcriptome.effects.cellcycle.get_cellcycle_genes()
sc.tl.score_genes_cell_cycle(
    adata,
    s_genes=cellcycle_genes.query("phase == 'S'")["gene"].tolist(),
    g2m_genes=cellcycle_genes.query("phase == 'G2/M'")["gene"].tolist(),
)

# %%
adata = adata[adata.obs["phase"] == "G1"].copy()

# %% [markdown]
# ## Datasets

# %% tags=["hide-input"]
import tempfile
import pathlib

project_root = pathlib.Path(tempfile.TemporaryDirectory().name)
project_root.mkdir()
laf.set_project_root(project_root)  # sets the default project root

# %% [markdown]
# _laflow_ contains two main classes that are used to construct datasets and models:

# %%
dataset = laf.Dataset("dataset")

# %%
dataset

# %% [markdown]
# A dataset contains one or more modalities. In single-cell data, we often use a {class}`laflow.dataset.MatrixModality`, or if we're working with transcriptome data, a {class}`lacell.transcriptome.TranscriptomeModality`:

# %%
modality = lac.transcriptome.TranscriptomeModality("transcriptome")

# %%
modality

# %%
dataset.add_modality(modality)

# %% [markdown]
# Dataset now has a transcriptome process, and the transcriptome was added to the "modalities" group:

# %%
dataset.transcriptome

# %%
dataset

# %%
dataset.transcriptome

# %%
dataset.transcriptome.from_adata(adata=adata)

# %%
dataset.transcriptome

# %% [markdown]
# ## Models

# %%
model = laf.Model("model", reset=True)

# %%
model.dataset = dataset

# %%
model.add_process(lac.transcriptome.Transcriptome())

# %%
model.add_process(lac.cell.batch.Batch())

# %%
model.add_process(lac.cell.gradient.GradientLatent(label="differentiation"))

# %%
model.root_initial.plot()

# %%
model.transcriptome.add_effect(
    lac.transcriptome.effects.gradient.GradientEffect(gradient=model.differentiation)
)

# %%
model.transcriptome.add_effect(
    lac.transcriptome.effects.batch.BatchEffect(batch=model.batch)
)

# %%
model.root_initial.plot()

# %%
model.infer()

# %%
model.differentiation.interpret()

# %%
model.transcriptome.differentiation.interpret()

# %%
adata.obs["differentiation"] = model.differentiation.observed.mean.to_pandas()

# %%
sc.pl.umap(adata, color="differentiation")

# %%
model.transcriptome.differentiation.causal.plot_features()

# %% [markdown]
# ## Making small manual adjustments

# %% [markdown]
# Imagine that you want to use these processes and effect, but just want to make one tiny adjustment. For example, in this case we may want to adjust the prior distribution as we did in the [manifolds](3-manifold) tutorials.

# %% [markdown]
# Let's first create the model as before:

# %%
model = laf.Model("model_manual_adjustments", dataset=dataset, reset=True)

model.add_process(lac.transcriptome.Transcriptome())

model.add_process(lac.cell.batch.Batch())
model.add_process(lac.cell.gradient.GradientLatent(label="differentiation"))

# %%
model.transcriptome.add_effect(
    lac.transcriptome.effects.gradient.GradientEffect(gradient=model.differentiation)
)
model.transcriptome.add_effect(
    lac.transcriptome.effects.batch.BatchEffect(batch=model.batch)
)

# %% [markdown]
# `model.root_initial` contains the full model, but can still be changed before inference. In this case, we want to adjust the prior of the differentiation latent variable.
#
# First, we have to extract this latent variable, and for this you have a couple of options. You can either manually use the the find function to recursively traverse the tree:
#
# ```python
# model.root_initial.find("differentiation")
# ```
#
# However, this can cause problems if the model contains named multiple variables with "differentiation" as label. A more direct way is to use the extract function of the process:

# %%
gradient = model.differentiation.extract(model.root_initial)

# %% [markdown]
# We can now adapt the prior distribution as before:

# %%
import pandas as pd

cells = gradient.value_definition["cell"]

gradient_p = la.distributions.Beta(
    beta=la.Fixed(
        pd.Series([1.0, 100.0], index=["Myod1", "mCherry"])[
            adata.obs["gene_overexpressed"]
        ].values,
        definition=[cells],
    ),
    alpha=la.Fixed(
        pd.Series([1.0, 1.0], index=["Myod1", "mCherry"])[
            adata.obs["gene_overexpressed"]
        ].values,
        definition=[cells],
    ),
)
gradient.p = gradient_p

# %%
gradient.plot()

# %%
model.infer()

# %%
model.differentiation.interpret()

# %%
model.transcriptome.differentiation.interpret()

# %%
adata.obs["differentiation"] = model.differentiation.observed.mean.to_pandas()

# %%
sc.pl.umap(adata, color="differentiation")

# %%
model.transcriptome.differentiation.causal.plot_features()


# %% [markdown]
# ## Custom processes

# %% [markdown]
# To make things easier to maintain and modularize, we can put our adaptations inside a subclass of {class}`lacell.cell.gradient.GradientLatent`. The

# %%
class DifferentiationLatent(lac.cell.gradient.GradientLatent):
    label = laf.PyObj(default=lambda: "differentiation")

    parent = lac.cell.gradient.GradientLatent.parent

    @lac.cell.gradient.GradientLatent.adapt.extend()
    def adapt(self, output, root_initial, obs, **kwargs):
        # call super adapt function
        output = super().adapt_(
            output=output, root_initial=root_initial, obs=obs, **kwargs
        )
        gradient = self.extract(root_initial)

        cells = gradient.value_definition["cell"]

        import pandas as pd

        gradient_p = la.distributions.Beta(
            beta=la.Fixed(
                pd.Series([1.0, 100.0], index=["Myod1", "mCherry"])[
                    obs["gene_overexpressed"]
                ].values,
                definition=[cells],
            ),
            alpha=la.Fixed(
                pd.Series([1.0, 1.0], index=["Myod1", "mCherry"])[
                    obs["gene_overexpressed"]
                ].values,
                definition=[cells],
            ),
        )
        gradient.p = gradient_p

        output.root_initial = root_initial

        return output


# %%
model = laf.Model("model_custom_process", dataset=dataset, reset=True)

# %%
model.add_process(lac.transcriptome.Transcriptome())

model.add_process(lac.cell.batch.Batch())
model.add_process(DifferentiationLatent())

# %%
model.transcriptome.add_effect(
    lac.transcriptome.effects.gradient.GradientEffect(gradient=model.differentiation)
)
model.transcriptome.add_effect(
    lac.transcriptome.effects.batch.BatchEffect(batch=model.batch)
)

# %%
model.root_initial.plot()

# %%
model.infer()

# %%
model.differentiation.interpret()

# %%
adata.obs["differentiation"] = model.differentiation.observed.mean.to_pandas()
sc.pl.umap(adata, color="differentiation")


# %% [markdown]
# ## Custom effects

# %% [markdown]
# {class}`~lac.transcriptome.effects.gradient.GradientEffect` uses a spline function by default, but maybe we want to use a simpler linear function:

# %%
class DifferentiationEffect(lac.transcriptome.effects.gradient.GradientEffect):
    @lac.transcriptome.effects.gradient.GradientEffect.adapt.extend()
    def adapt(self, output, root_initial, **kwargs):
        # extract gradient and foldchange
        gradient = self.gradient.extract(root_initial)
        foldchange = self.parent.extract_foldchange(root_initial)

        # add effect to foldchange
        effect = la.links.scalar.Switch(
            gradient, a=True, shift=True, definition=foldchange.value_definition
        )
        setattr(foldchange, self.label, effect)

        output.root_initial = root_initial

        return output


# %%
model = laf.Model("model_custom_effect", dataset=dataset, reset=True)

# %%
model.add_process(lac.transcriptome.Transcriptome())

model.add_process(lac.cell.batch.Batch())
model.add_process(DifferentiationLatent())

# %%
model.transcriptome.add_effect(DifferentiationEffect(gradient=model.differentiation))
model.transcriptome.add_effect(
    lac.transcriptome.effects.batch.BatchEffect(batch=model.batch)
)

# %%
model.root_initial.plot()

# %%
model.infer()

# %%
model.differentiation.interpret()

# %%
adata.obs["differentiation"] = model.differentiation.observed.mean.to_pandas()
sc.pl.umap(adata, color="differentiation")

# %%
model.transcriptome.differentiation.interpret()

# %%
model.transcriptome.differentiation.causal.plot_features()

# %%

# %%

# %%
