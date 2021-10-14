# %%
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        pass
    else:
        print("running from IPython")
        import os

        import laflow

        os.chdir(laflow.get_project_root())
except:
    # We do not even have IPython installed
    pass


# %%
import os
from pathlib import Path
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# %%
cache_file = Path(".nb-cache")
if not cache_file.exists():
    cache = {}
else:
    with cache_file.open("r") as f:
        cache = json.load(f)

# %%
notebooks = []
for path in Path("_book").rglob("./*.ipynb"):
    if ".ipynb_checkpoints" in str(path):
        continue
    notebooks.append(path)

# %%
skip = False
if skip:
    for notebook in notebooks:
        cache[str(notebook)] = os.path.getmtime(notebook)

# %%
dry = False
for notebook in notebooks:
    if str(notebook) not in cache:
        cache[str(notebook)] = None

    if (cache[str(notebook)] is None) or (
        os.path.getmtime(notebook) != cache[str(notebook)]
    ):
        print(notebook)
        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            if not dry:
                ep.preprocess(nb)
                with open(notebook, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)
                cache[str(notebook)] = os.path.getmtime(notebook)
        except Exception as e:
            print("expected")
            print(e)


# %%
with open(".nb-cache", "w") as f:
    json.dump(cache, f)

# %%
