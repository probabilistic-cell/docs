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
# # Dynamics

# %%
import pandas as pd

# %% tags=["remove-input"]
from IPython.display import display, Markdown, Latex, HTML
cookbooks = pd.DataFrame([
    [
        "transcription",
        "A simple transcription"
    ]
], columns = ["id", "description"]).set_index("id")
cookbooks["url"] = [f"<a href='{id}.html'>{id}</a>" for id in cookbooks.index] 
html = cookbooks.to_html(index = False, header = False, escape = False)

html = '<div class="admonition note" name="html-admonition"><p>' + html + '</p></div>'

display(HTML(html))


# %% [markdown]
#
