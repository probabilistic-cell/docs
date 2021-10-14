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
# # Simple gene regulatory networks

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import torch

import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import collections

import latenta as la
import ladynamic as lad

# %%
import networkx as nx

# %%
genes = la.Dim(5, "gene")
proteins = la.Dim(range(len(genes)), "protein")

# %%
mrna_initial = la.Fixed(1.0)
protein_initial = la.Fixed(0.0)
transcription_maximal = la.Fixed(
    np.array([0.0, 10.0, 20.0, 30.0, 40.0]), definition=la.Definition([genes])
)
degradation = la.Fixed(
    np.array([1.0, 1.0, 1.0, 1.0, 1.0]), definition=la.Definition([genes])
)

# %%
mrna = lad.State(mrna_initial, definition=la.Definition([genes]), label="mrna")
protein = lad.State(mrna_initial, definition=la.Definition([genes]), label="protein")


# %%
class dmrna(la.Computed):
    transcription_maximal = la.variables.VariablePromise(
        "transcription_maximal", r"\mathit{transcription}_{max}"
    )
    degradation = la.variables.VariablePromise("degradation")
    mrna = la.variables.VariablePromise("mrna")

    def __call__(self, transcription_maximal, degradation, mrna):
        return transcription_maximal - degradation * mrna


class dprotein(la.Computed):
    mrna = la.variables.VariablePromise(
        dims=la.variables.promise.ParentSwitched("transcript_dim", "protein_dim")
    )
    protein = la.variables.VariablePromise(
        dims=la.variables.promise.ParentSwitched("protein_dim", "transcript_dim")
    )

    def __init__(self, mrna, protein, *args, definition=None, **kwargs):
        self.transcript_dim = mrna[-1]
        self.protein_dim = protein[-1]

        definition = mrna.clean.contract(mrna[-1]).expand_right(protein[-1])
        super().__init__(*args, definition=definition, **kwargs)

    def __call__(self, mrna, protein):
        return protein - mrna


# %%
mrna.derivative = dmrna(transcription_maximal, degradation, mrna)
protein.derivative = dprotein(mrna, protein)

# %%
mrna.plot()

# %%
dt = la.Fixed(0.01)
final_time = la.Fixed(10.0)
delta_time = la.Fixed(1.0)
mrna_extractor = lad.extractors.DeltaExtractor(mrna, final_time, delta_time)

extractors = [mrna_extractor]

# %%
states = [mrna]
dynamics = lad.solvers.Euler(states, extractors, dt)

# %%
time = dynamics.time
time_extractor = lad.extractors.DeltaExtractor(time, final_time, delta_time)
dynamics.add_extractors(time_extractor)

# %%
dispersion = la.Fixed(0.01)
counts = la.distributions.NegativeBinomial2(mrna_extractor, dispersion)

# %%
model = la.Root(counts=counts, time=time_extractor)
model.plot()

# %%
posterior = la.posterior.Posterior(
    model,
    batch_dim="",
    retain_samples={time_extractor, mrna_extractor, counts, counts.mu},
)
posterior.sample(10)

# %% tags=[]
time_values = (
    posterior.samples[time_extractor]
    .sel(sample=0)
    .broadcast_like(posterior.samples[mrna_extractor].sel(sample=0))
)
steps = xr.concat(
    [
        posterior.samples[counts].sel(sample=0),
        posterior.samples[counts.mu].sel(sample=0),
        time_values,
    ],
    dim=pd.Series(["counts", "expression", "time"], name="state"),
)

# %%
sns.set_palette("Set1", 8, 0.75)

# %%
import matplotlib as mpl

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
for gene_id in genes.index:
    plotdata = steps.sel(gene=gene_id).to_pandas().T
    ax.scatter(plotdata["time"], plotdata["counts"], s=1)
    ax.plot(plotdata["time"], plotdata["expression"])

# %%
last_step = posterior.samples[counts].coords["step"][-1]
counts_value = posterior.samples[counts].sel(step=last_step).to_pandas()

# %%
counts_value.index = pd.Series(counts_value.index, name="cell")

# %% [markdown] tags=[]
# ## Optimizing dynamic parameters

# %%
mrna_initial = la.Fixed(1.0)
# transcription_maximal = la.Parameter(10., definition = la.Definition([genes]), transforms=[la.transforms.Exp()])
transcription_maximal = la.Latent(
    la.distributions.LogNormal(), definition=la.Definition([genes])
)
# transcription_maximal = la.Fixed(10., definition = la.Definition([genes]), transforms=[la.transforms.Exp()])
degradation = la.Fixed(
    np.array([1.0, 1.0, 1.0, 1.0, 1.0]), definition=la.Definition([genes])
)

# %%
mrna = lad.State(mrna_initial, definition=la.Definition([genes]), label="mrna")

# %%
mrna.derivative = dmrna(transcription_maximal, degradation, mrna)

# %%
mrna_extractor = lad.extractors.FinalExtractor(mrna, final_time)

extractors = [mrna_extractor]

# %%
states = [mrna]
dynamics = lad.solvers.Euler(states, extractors, dt)

# %%
counts_p = la.distributions.NegativeBinomial2(
    mrna_extractor, dispersion, definition=la.Definition.from_xr(counts_value)
)

# %%
counts = la.Observation(counts_value, counts_p, label="counts")

# %%
model = la.Root(counts)
model.plot()

# %%
inference = la.infer.svi.SVI(
    model, [la.infer.loss.ELBO()], la.infer.optim.Adam(lr=0.05)
)
trainer = la.infer.trainer.Trainer(inference)
trace = trainer.train(1000)
trace.plot()

# %%
posterior = la.posterior.Posterior(model)
posterior.sample(10)

# %%
