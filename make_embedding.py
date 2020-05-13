# test proximity
from os.path import join
import git
# from openne import graphrep
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import networkx as nx

from toolbox import wrappers, network_utilities
from utils import load_drugs_from, load_diseases_from
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ppi', type=str, default='STRING',
                    choices=['STRING'],
                    help='Which PPI data to use,')
parser.add_argument('--dti', type=str, default='Drugbank',
                    choices=['Drugbank'],
                    help='Which DTI data to use,')
parser.add_argument('--mode', type=str, default='Covid-19-train',
                    choices=['Covid-19-train'],
                    help='Which mode to use')
parser.add_argument('--embs', type=str, default='GraphRep',
                    help='Which embs method to use')
parser.add_argument('--gcn', type=str, default='gcn',
                    help='Which gcn model to use and whether use a gcn after embs.'
                    ' Default: gcn. Set to None if not using gcn ')
args = parser.parse_args()

# ==================================================
# -- load data
# PPI graph
if args.ppi == 'STRING':
    G = nx.read_edgelist(
        "resources/BioNEV/data/STRING_PPI/STRING_PPI.edgelist")
    nodes = pd.read_csv(
        'resources/BioNEV/data/STRING_PPI/node_list.txt', sep='\t', index_col=0)
else:
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
file_name = f"{args.ppi}_PPI_{args.embs}_embs.txt"
emb_folder = 'saved/embs'
ppi_embs = np.loadtxt(join(emb_folder, file_name), skiprows=1)
ppi_id = ppi_embs[:, 0].astype(int)
ppi_embs = ppi_embs[:, 1:]
# ==================================================

# ==================================================
# -- train the gcn
# ==================================================

# ==================================================
# -- store model
# ==================================================

# ==================================================
# -- make predictions
# ==================================================

# validation

# ==================================================
# -- save file
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
script_name = repo.git.rev_parse(sha, short=6) + '.py'
# ==================================================
