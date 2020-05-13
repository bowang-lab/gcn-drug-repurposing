# test proximity
import warnings
from os.path import join
import git
# from openne import graphrep
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import networkx as nx

from toolbox import wrappers, network_utilities
from utils import load_drugs_from, load_diseases_from, load_DTI, make_SARSCOV2_PPI
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
    drug_target_dict = load_DTI()
    covid_protein_list = make_SARSCOV2_PPI()
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
protein_embs_dict = {}
for i, id in enumerate(ppi_id):
    protein_embs_dict[nodes.at[id, 'STRING_id']] = ppi_embs[i]
# ==================================================

# ==================================================
# -- train the gcn
# ==================================================

# ==================================================
# -- store model
# ==================================================

# ==================================================
# -- make predictions

# ppid2id
nodes['id'] = nodes.index
ppid2id = nodes.set_index('STRING_id').to_dict()['id']  # len 15131

# covid_emb
covid_emb = []
for protein in covid_protein_list:
    try:
        covid_emb.append(protein_embs_dict[protein])
    except:
        print(f'missing protein {protein} in ppi data')
covid_emb = np.vstack(covid_emb)  # (238, 100)

# FIXME: the number of embeddings are not consistent!!!

# drugs_embs
drugs_orders = []
drug_emb_matrix = []
for k, v in drug_target_dict.items():
    tmp = []
    for protein in v:
        try:
            tmp.append(protein_embs_dict[protein])
        except:
            print(f'missing protein {protein} in ppi data')
    if len(tmp) > 0:
        drugs_orders.append(k)
        drug_emb_matrix.append(np.vstack(tmp).mean(0))
    else:
        print(f'omit drug {k} with targets {v}\n')
drug_emb_matrix = np.vstack(drug_emb_matrix)

# ranks of drugs
ranks = np.argsort((drug_emb_matrix @ covid_emb.T).max(1))[
    ::-1]  # ranks of drugs
# ==================================================

# validation

# ==================================================
# -- save file
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
script_name = repo.git.rev_parse(sha, short=6) + '.py'
# ==================================================
