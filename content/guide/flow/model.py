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

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
import latenta as la
import laflow as laf

import numpy as np

# %%
import tempfile

tmpdir = tempfile.TemporaryDirectory().name

model = laf.Model("model", project_root=tmpdir)

# %%
cells = la.Dim(3, "cell")
genes = la.Dim(4, "gene")

dataset = laf.Dataset("dataset", project_root=tmpdir)
dataset.add_modality(
    laf.dataset.MatrixModality(
        "transcriptome",
        dataset,
        X=np.random.random(size=(cells.size, genes.size)),
        obs=cells.index.to_frame().set_index("cell"),
        var=genes.index.to_frame().set_index("gene"),
        reset=True,
    )
)

# %%
model = laf.Model(dataset=dataset, name="", project_root=tmpdir)

# %%
model.root_initial = la.Latent(
    p=la.distributions.Normal(la.Parameter(1.0)),
    q=la.distributions.Normal(-1.0, definition=[cells, genes]),
    label="latent",
)

# %%
model.root_initial.plot()

# %%
dataset.test = model.root_

# %%
dataset.test_

# %%
model

# %%
model.infer()

# %%
model.trace.plot()

# %%
posterior = la.posterior.scalar.ScalarPredictiveve(model.root.q)
posterior.sample(1)

# %% [markdown]
# As explained in [], {class}`laflow.Latenta`, you can provide a `db` which will avoid that any components stored in the posterior (in this case the model.root_) will be saved multiple times.

# %%
model.posterior = laf.Latenta(db={model.root_})
model.posterior = posterior

# %%
# ! ls -lh {model.path}/*.pkl

# %%
ax = model.posterior.plot()
model.q_distribution = ax

# %%
model

# %%
