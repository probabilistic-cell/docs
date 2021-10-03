# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--production",
    help="Whether in production or not.",
    action="store_true",
    default=False,
)
args = parser.parse_args()

import os

if not args.production:
    os.chdir("../")

# %%
from pathlib import Path
import nbformat as nbf
from jupyter_cache import get_cache
from jupyter_cache.base import NbBundleIn
from jupyter_cache.executors import load_executor, list_executors
from jupyter_cache.utils import tabulate_cache_records, tabulate_stage_records

# %%
cache = get_cache(".jupyter_cache")
cache

# %%
notebooks = []
for path in Path("_book").rglob("./*.ipynb"):
    if ".ipynb_checkpoints" in str(path):
        continue
    notebooks.append(path)

# %%
skip = False
if skip:
    for path in notebooks:
        record = cache.cache_notebook_file(
            path=path, check_validity=False, overwrite=True
        )
        record

# %%
for path in notebooks:
    record = cache.stage_notebook_file(path=path)
    record

# %%
print(cache.list_staged_unexecuted())
executor = load_executor("basic", cache=cache)
executor

# %%
result = executor.run_and_cache(timeout=180)
result

# %%
for record in cache.list_staged_unexecuted():
    print(record.traceback)
# for record in cache.list_staged_


# %%
