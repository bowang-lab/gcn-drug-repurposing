from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles

# Construct the multiscale interactome
msi = MSI(drug2protein_file_path="data/drug_to_protein.tsv",
          drug2protein_directed=False)
