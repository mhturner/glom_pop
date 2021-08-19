# %%

import pandas as pd
from scipy.io import mmread
import os

data_dir = '/Users/mhturner/Downloads'


# %%
a = mmread(os.path.join(data_dir, 'GSE156455_matrix_main.mtx.gz'))
a.shape
# %%
import scprep
data = scprep.io.load_10X(data_dir)

data = scprep.filter.remove_empty_cells(data)
data = scprep.filter.remove_empty_genes(data)

# %%
