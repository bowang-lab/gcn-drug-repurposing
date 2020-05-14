# test proximity
from matplotlib import pyplot as plt
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
# normalize

# ranks of drugs
ranks = np.argsort((drug_emb_matrix @ covid_emb.T).max(1))[
    ::-1]  # ranks of drugs
# ==================================================

# ==================================================
# -- validation
drugs_in_trial = pd.read_csv('data/Covid-19 Clinical Trials.csv')
union_drugs_in_trial = []
drugs_orders_label = np.zeros(len(drugs_orders), dtype=int)
for id in drugs_in_trial['DrugBank ID']:
    if id in drugs_orders:
        union_drugs_in_trial.append(id)
        drugs_orders_label[drugs_orders.index(id)] = 1
print(f'total {len(union_drugs_in_trial)} experimental Covid-19 drugs found in the current drug gallery')

num_pos = len(union_drugs_in_trial)
hit_cnt = 0
recall_curve = np.zeros(len(ranks), dtype=float)
for i, id in enumerate(ranks):
    if drugs_orders[id] in union_drugs_in_trial:
        hit_cnt += 1
        print(f"the {i}-th proposed drug is in the clinical trial list. " +
              f"found {hit_cnt} clinical trial drugs." +
              f"recall {hit_cnt/num_pos}.")
    recall_curve[i] = hit_cnt/num_pos
# plot
plt.figure()
lw = 2
plt.plot(recall_curve, color='darkorange',
         lw=lw, label='Recall1')
plt.plot([0, len(ranks)], [0, 1], color='navy', lw=lw, linestyle='--')
plt.ylim([0.0, 1.05])
plt.xlabel('# in ranks')
plt.ylabel('Recall')
plt.title('Recall of drugs undergoing clinical trials')
plt.legend(loc="lower right")
plt.savefig('saved/figures/recall.png')
# ==================================================

# ==================================================
# -- save file
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
script_name = repo.git.rev_parse(sha, short=6) + '.py'
# ==================================================
