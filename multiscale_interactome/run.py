from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles


# Construct the multiscale interactome
msi = MSI()
msi.load()

# Test against reference
# teselst_msi()

# Calculate diffusion profiles
print('Calculate diffusion profiles')
dp = DiffusionProfiles(
    alpha=0.8595436247434408,
    max_iter=1000,
    tol=1e-06,
    weights={
        'down_functional_pathway': 4.4863053901688685,
        'indication': 3.541889556309463,
        'functional_pathway': 6.583155399238509,
        'up_functional_pathway': 2.09685000906964,
        'protein': 4.396695660380823,
        'drug': 3.2071696595616364
    },
    num_cores=int(multiprocessing.cpu_count()/2),
    save_load_file_path="results/"
)
dp.calculate_diffusion_profiles(msi)

# Load saved diffusion profiles
dp_saved = DiffusionProfiles(
    alpha=None,
    max_iter=None,
    tol=None,
    weights=None,
    num_cores=None,
    save_load_file_path="results/"
)
msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)


# Diffusion profile for Rosuvastatin (DB01098)
dp_saved.drug_or_indication2diffusion_profile["DB01098"]


# Test against reference
test_diffusion_profiles("data/top_msi/", "results/")
