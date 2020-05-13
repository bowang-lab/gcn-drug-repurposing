# test proximity
from openne import graphrep
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from toolbox import wrappers, network_utilities
from utils import load_drugs_from, load_diseases_from
import time

# ==================================================
# -- load data
# PPI graph
network_file = "2016data/network/network.sif"
G = wrappers.get_network(network_file, only_lcc=True)

# provide drugs objects
drugs = load_drugs_from("2016data/target/drug_to_geneids.pcl.all")

# provide disease objects
diseases = load_diseases_from("2016data/disease/disease_genes.tsv")
# ==================================================

# ==================================================
# -- graph embedding - graphrep
# try loading pretrained embedding
# ==================================================

# ==================================================
# -- train the gcn
# ==================================================

# ==================================================
# -- store model
# ==================================================

# make predictions

# validation
