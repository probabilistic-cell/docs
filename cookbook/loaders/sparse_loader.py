# %%
from IPython import get_ipython
if get_ipython():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import latenta as la
import scanpy as sc
import numpy as np
import torch


# %%
adata = sc.datasets.pbmc3k()


# %%
device = "cuda"
# device = "cpu"


# %%
import scipy.sparse

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_scipy = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))

coo = la.sparse.COO.from_scipy_csr(csr_scipy, device = "cpu")
coo.populate_mapping()


# %%
# %%timeit
coo.dense_subset(np.array([0, 1, 2]))


# %%
import scipy.sparse


# %%
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_scipy = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))

coo = la.sparse.COO.from_scipy_csr(csr_scipy, device = "cpu")
coo.populate_row_switch()
coo.populate_mapping()


# %%
ix = np.random.choice(coo.shape[0], 1000)
coo.dense_subset(ix)


# %%
# %%timeit
coo.dense_subset(ix)


# %%
ix = slice(1, 1000)
ix_sparse = slice(shifts[ix.start], shifts[ix.stop])
indices_selected = out._indices()[:, ix_sparse].clone()
indices_selected[0, :] = indices_selected[0, :] - ix.start
values_selected = out._values()[ix_sparse]

subsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (ix.stop - ix.start, *out.shape[1:])).to_dense()


# %%
# %%prun
ix = slice(1, 1000)
ix_sparse = slice(shifts[ix.start], shifts[ix.stop])
indices_selected = out._indices()[:, ix_sparse].clone()
indices_selected[0, :] = indices_selected[0, :] - ix.start
values_selected = out._values()[ix_sparse]

subsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (ix.stop - ix.start, *out.shape[1:])).to_dense()


# %%
get_ipython().run_cell_magic('prun', '', 'ix = np.random.choice(out.shape[0], 1000)\nix_sparse = torch.cat([torch.arange(shifts[row], shifts[row+1], device = device) for i, row in enumerate(ix)])\nrow_index_adjustments = torch.cat([torch.tensor(-row + i, device = device).repeat(shifts[row+1] - shifts[row]) for i, row in enumerate(ix)])\nindices_selected = out._indices()[:, ix_sparse].clone()\nindices_selected[0, :] = indices_selected[0, :] + row_index_adjustments\nvalues_selected = out._values()[ix_sparse]\n\nsubsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (len(ix), *out.shape[1:])).to_dense()')


# %%
get_ipython().run_cell_magic('prun', '', 'ix = np.random.choice(out.shape[0], 1000)\n\nindices = []\nvalues = []\n\n_indices = out._indices()#.clone()\n\nfor i, row in enumerate(ix):\n    indices_selected = torch.stack((\n        torch.tensor(i, device = device).repeat(shifts[row+1] - shifts[row]),\n        _indices[1, shifts[row]:shifts[row+1]]\n    ))\n    indices.append(indices_selected)\n    values.append(out._values()[shifts[row]:shifts[row+1]])\nindices_selected = torch.cat(indices, 1)\nvalues_selected = torch.cat(values)\n\nsubsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (len(ix), *out.shape[1:])).to_dense()')


# %%
get_ipython().run_cell_magic('prun', '', 'ix = np.random.choice(out.shape[0], 1000)\n\nindices = []\nvalues = []\n\n_indices = out._indices()#.clone()\n\nfor i, row in enumerate(ix):\n    indices_selected = torch.zeros((2, shifts[row+1] - shifts[row]), device = device, dtype = torch.long)\n    indices_selected[0, :] = i\n    indices_selected[1, :] = _indices[1, shifts[row]:shifts[row+1]]\n    \n    indices.append(indices_selected)\n    values.append(out._values()[shifts[row]:shifts[row+1]])\nindices_selected = torch.cat(indices, 1)\nvalues_selected = torch.cat(values)\n\nsubsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (len(ix), *out.shape[1:])).to_dense()')


# %%
get_ipython().run_line_magic('load_ext', 'line_profiler')


# %%
ix = np.random.choice(out.shape[0], 1000)

indices = []
values = []

_indices = out._indices()

new_shifts = torch.cat([torch.tensor([0], device = device), torch.cumsum(shifts[ix + 1] - shifts[ix], 0)])

indices_selected = torch.zeros((2, new_shifts[-1]), device = device, dtype = torch.long)

for i, row in enumerate(ix):
    indices_selected[0, new_shifts[i]:new_shifts[i+1]] = i
    indices_selected[1, new_shifts[i]:new_shifts[i+1]] = _indices[1, shifts[row]:shifts[row+1]]
    
    values.append(out._values()[shifts[row]:shifts[row+1]])
values_selected = torch.cat(values)

subsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (len(ix), *out.shape[1:])).to_dense()


# %%
def run(out, shifts):
    ix = np.random.choice(out.shape[0], 1000)

    indices = []
    values = []

    _indices = out._indices()

    new_shifts = torch.cat([torch.tensor([0], device = device), torch.cumsum(shifts[ix + 1] - shifts[ix], 0)])

    indices_selected = torch.zeros((2, new_shifts[-1]), device = device, dtype = torch.long)
    
    values_selected = torch.zeros((new_shifts[-1], ), device = device, dtype = out.dtype)

    for i, row in enumerate(ix):
        indices_selected[0, new_shifts[i]:new_shifts[i+1]] = i
        indices_selected[1, new_shifts[i]:new_shifts[i+1]] = _indices[1, shifts[row]:shifts[row+1]]

        values_selected[new_shifts[i]:new_shifts[i+1]] = out._values()[shifts[row]:shifts[row+1]]

    subsetted = torch.sparse.FloatTensor(indices_selected, values_selected, (len(ix), *out.shape[1:])).to_dense()
    return subsetted


# %%
get_ipython().run_line_magic('lprun', '-f run run(out, shifts)')


# %%
get_ipython().run_cell_magic('prun', '', 'ix = np.random.choice(out.shape[0], 2000)\n\n_indices = out._indices()\n\n# ix_sparse = torch.cat([torch.arange(shifts[row], shifts[row+1], device = device) for row in ix]) # this is extremely slow, new tensory for each index...\nix_sparse = torch.cat([mapping[row] for row in ix])\nnewrow_idx = torch.arange(len(ix), device = device).repeat_interleave(shifts[ix+1] - shifts[ix])\nnewcol_idx = _indices[1, ix_sparse]\n\nsubsetted = torch.empty((len(ix), out.shape[1]), device = device, dtype = out.dtype)\n\nsubsetted[newrow_idx, newcol_idx] = out._values()[ix_sparse]')


# %%
def run(out, mapping):
    ix = np.random.choice(out.shape[0], 1000)
    
    _indices = out._indices()

    # ix_sparse = torch.cat([torch.arange(shifts[row], shifts[row+1], device = device) for row in ix]) # this is extremely slow, new tensory for each index...
    ix_sparse = torch.cat([mapping[row] for row in ix])
    newrow_idx = torch.arange(len(ix), device = device).repeat_interleave(shifts[ix+1] - shifts[ix])
    newcol_idx = _indices[1, ix_sparse]

    subsetted = torch.empty((len(ix), out.shape[1]), device = device, dtype = out.dtype)

    subsetted[newrow_idx, newcol_idx] = out._values()[ix_sparse]
    return subsetted
run(out, mapping)


# %%
get_ipython().run_cell_magic('timeit', '', 'run(out, mapping)')


# %%
import datetime
n = datetime.datetime.now()
run(out, mapping)
print(datetime.datetime.now() - n)


# %%
subsetted[newrow_idx, newcol_idx] = out._values()[ix_sparse]


# %%
subsetted.shape


# %%
subsetted


# %%
values_selected.shape


# %%
row_index_adjustments


# %%
indices_selected


# %%
indices_selected = out._indices()[:, ix_sparse].clone()


# %%
indices_selected


# %%
ix_sparse


# %%
indices_selected


# %%
ix = [1, 4, 6]


# %%



# %%
subsetted


# %%
subsetted.shape


