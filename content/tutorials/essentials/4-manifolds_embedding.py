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
# # Manifolds: embedding

# %%
# %load_ext autoreload
# %autoreload 2
import latenta as la
import lacell as lac

import numpy as np

# %% [markdown]
# We'll use the same dataset as [before](./1-variables)

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

# %% [markdown]
# ## Building an embedding model from scratch

# %%
adata_oi = adata

# %%
transcriptome = lac.transcriptome.create_observation_from_adata(adata_oi)

# %%
components = la.Dim(20, "component")
embedding = la.Latent(
    la.distributions.Normal(),
    definition=[transcriptome["cell"], components],
    label="embedding",
)

# %% [markdown]
# Now that we have defined the cellular latent space, we still have to define how this space affects the transcriptome. Given that we (can) assume that _most_ genes will change smoothly but non-linearly during differentiation, we choose a spline function for this:

# %%
amortization_input = la.Fixed(transcriptome.loader, definition=transcriptome)

# %%
encoder = la.amortization.Encoder(
    amortization_input, embedding
)
embedding.plot()

# %% [markdown]
# We lose interpretability, but that's a price we have to pay for flexibility.

# %%
foldchange = transcriptome.find("foldchange")

foldchange.embedding = la.links.nn.NN.create_feedforward(embedding, foldchange)

# %%
foldchange.embedding.nn

# %% [markdown]
# As before, we also want to correct for the batch effect (again remember that the foldchange is an additive modular variable, so we can add any components we want and the foldchange will be a linear combination of them):

# %%
batch = la.variables.discrete.DiscreteFixed(adata_oi.obs["batch"])
foldchange.batch = la.links.vector.Matmul(batch, definition=foldchange.value_definition)

# %%
foldchange.plot()

# %%
with transcriptome.switch(la.config.device):
    inference = la.infer.svi.SVI(
        transcriptome, [la.infer.loss.ELBO()], optimizer = la.infer.optim.Adam()
    )
    trainer = la.infer.trainer.Trainer(inference)
    trace = trainer.train(10000)
    trace.plot()

# %% [markdown]
# We can extract the inferred values using a {class}`~latenta.posterior.scalar.ScalarPredictive` posterior:

# %%
embedding_predictive = la.posterior.vector.VectorPredictive(embedding)
embedding_predictive.sample(10)

# %%
embedding_predictive.plot()

# %%
mean = embedding_predictive.mean.to_pandas()
adata_oi.obsm["embedding"] = mean
sc.pp.neighbors(adata_oi, use_rep = "embedding")
sc.tl.umap(adata_oi)

# %%
sc.pl.umap(adata_oi, color=["gene_overexpressed", "batch", "phase"])

# %% [markdown]
# ## Building an embedding model using `lacell`

# %%
import laflow as laf

# %%
flow = laf.TemporaryFlow()

# %%
dataset = lac.transcriptome.create_dataset_from_adata(adata, "dataset", flow)
model = lac.Model("model", flow, dataset = dataset)

# %%
model.add_process(lac.transcriptome.Transcriptome())

# %%
model.add_process(lac.cell.embedding.Latent())

# %%
model.transcriptome.add_effect(lac.transcriptome.effects.embedding.Decoder(embedding = model.embedding))

# %%
model.add_process(lac.cell.batch.Given())

# %%
model.transcriptome.add_effect(lac.transcriptome.effects.batch.Linear(batch = model.batch))

# %%
model.infer()

# %%
model.embedding.extract_twod_representation()

# %%
adata.obsm["X_umap"] = model.embedding.twod_representation

# %%
adata.obsm["X_umap"]

# %%
sc.pl.umap(adata_oi, color=["gene_overexpressed", "batch", "phase"])
