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

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %%
import latenta as la
import laflow as laf

# %%
import tempfile

tmpdir = tempfile.TemporaryDirectory().name

model = laf.Model(project_root=tmpdir)

# %%
model.model_initial = la.Latent(
    p=la.distributions.Normal(la.Parameter(1.0)),
    q=la.distributions.Normal(-1.0, definition=[la.Dim(500, "dim")]),
    label="latent",
)

# %%
model.model_initial.plot()

# %%
model

# %%
model.infer()

assert model.model_.available()

# %%
model

# %%
model.trace.plot()

# %%
posterior = la.posterior.scalar.ScalarObserved(model.model.q)
posterior.sample(1)

# %% [markdown]
# As explained in [], {class}`laflow.Latenta`, you can provide a `db` which will avoid that any components stored in the posterior (in this case the model.model_) will be saved multiple times.

# %%
model.posterior = laf.Latenta(db={model.model_})
model.posterior = posterior

# %%
# ! ls -lh {model.path}/*.pkl

# %%
ax = model.posterior.plot()

# %%
model.q_distribution = ax

# %%
model

# %%

# %%

# %%
