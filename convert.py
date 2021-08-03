# %%
import os
import jupytext

# %%
from pathlib import Path

path = "./tutorials/essentials/1-variables.py"
py = open(path, "r").read()

def convert(py):
    return py

for path in Path('tutorials').rglob('./*.py'):
    if ".ipynb_checkpoints" in str(path):
        continue

    py = open(path, "r").read()
    py = convert(py)

    nb = jupytext.reads(py, fmt = "py")

    jupytext.write(nb, str(path.with_suffix(".ipynb")))
    
# %%
