# %%
import os
import jupytext
# os.chdir("../")

# %%
from pathlib import Path

for path in Path('content').rglob('./*.ipynb'):
    if ".ipynb_checkpoints" in str(path):
        continue
    pypath = path.with_suffix(".py")
    if os.path.exists(str(pypath)):
        if abs(os.path.getmtime(path) - os.path.getmtime(pypath)) > 10:
            # print(path)
            path.touch()
            pypath.touch()
            os.system(f"jupytext --sync {path}")
    
# %%
