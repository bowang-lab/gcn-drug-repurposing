# %run make_embedding.py --gcn None --embs GraphRep
from matplotlib import pyplot as plt
import warnings
from os.path import join, exists
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
parser.add_argument('--embs', type=str, default='struc2vec',
                    choices=['GraphRep', 'node2vec', 'z-score'],
                    help='Which embs method to use')
parser.add_argument('--gcn', type=str, default='gcn',
                    help='Which gcn model to use and whether use a gcn after embs.'
                    ' Default: gcn. Set to None if not using gcn ')
parser.add_argument('--beam', type=int, default=5,
                    help='beam size')
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
    # ppid2id
    nodes['id'] = nodes.index
    ppid2id = nodes.set_index('STRING_id').to_dict()['id']  # len 15131
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
if args.embs == 'struc2vec':
    if exists(join(emb_folder, file_name)):
        ppi_embs = np.loadtxt(join(emb_folder, file_name), skiprows=1)
        ppi_id = ppi_embs[:, 0].astype(int)
        ppi_embs = ppi_embs[:, 1:]
        protein_embs_dict = {}
        for i, id in enumerate(ppi_id):
            protein_embs_dict[nodes.at[id, 'STRING_id']] = ppi_embs[i]
elif args.embs == 'z-score':
    pass
# ==================================================

# ==================================================
# -- train the gcn
if args.gcn:
    print('using gcn')
# ==================================================

# ==================================================
# -- store model
# ==================================================

# ==================================================
# -- make predictions using z-score


def make_nodes(list_of_protein_name):
    nodes = [str(ppid2id.get(name))
             for name in list_of_protein_name if name in ppid2id]
    return nodes


drugs_orders = []
if args.embs == 'z-score':
    drug_dists = []
    cnt = 0
    covid_nodes = make_nodes(covid_protein_list)
    for k, v in drug_target_dict.items():
        cnt += 1
        print(cnt)
        start_time = time.time()
        drug_nodes = make_nodes(v)
        if drug_nodes:
            try:
                tmp = wrappers.calculate_closest_distance(
                    G, nodes_from=drug_nodes, nodes_to=covid_nodes)
                drug_orders.append(k)
                drug_dists.append(tmp)
            except:
                pass
    drug_dists = np.array(drug_dists)
    ranks = np.argsort(drug_dists)
# ==================================================


# ==================================================
# -- make predictions
if args.embs == 'struc2vec':
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

drug_rank = []
for i in ranks:
    drug_rank.append(drugs_orders[i])
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

# ==================================================
# -- combination
# computing using covid_emb and drug_emb_matrix

# given a rank of drugs, building the beam search
beam_size = args.beam
# ==================================================
