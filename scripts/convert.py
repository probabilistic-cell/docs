# %%
# When running from IPython
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        pass
    else:
        print("running from IPython")
        import os

        import laflow

        os.chdir(laflow.get_git_root())
except:
    # We do not even have IPython installed
    pass

import os
import jupytext

# %%
from pathlib import Path

for path in Path("content").rglob("./*.ipynb"):
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
