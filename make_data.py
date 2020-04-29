#%% load PPI data
import csv
import json
import numpy as np
from scipy.sparse import coo_matrix

#%% load the full link data
# PPI_path = '/scratch/hdd001/home/haotian/Covid19Datasets/PPI/9606.protein.links.v11.0.txt'
# PPI = []
# ppid2index = {}
# index2ppid = {}
# cnt = 0
# with open(PPI_path, 'r') as tsvfile:
#     reader = csv.reader(tsvfile, delimiter=' ')
#     header = reader.__next__()
#     for i, row in enumerate(reader):
#         if not row[0] in ppid2index:
#             ppid2index[row[0]] = cnt
#             index2ppid[cnt] = row[0]
#             cnt += 1
#         if not row[1] in ppid2index:
#             ppid2index[row[1]] = cnt
#             index2ppid[cnt] = row[1]
#             cnt += 1
#         PPI.append([ppid2index[row[0]], ppid2index[row[1]], int(row[2])])
#         print(i)
#         # if i > 10000:
#         #     break
# # print(f"{PPI}\n{ppid2index}\n{index2ppid}")
# print(f"total count {cnt}")

#%% load action link data
PPI_path = '/scratch/hdd001/home/haotian/Covid19Datasets/PPI/9606.protein.actions.v11.0.txt'
PPI = [[0,0,0]]
ppid2index = {}
index2ppid = {}
cnt = 0
with open(PPI_path, 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    header = reader.__next__()
    print(header)
    for i, row in enumerate(reader):
        if row[2] in ['activation', 'binding', 'catalysis', 'reaction']:
            ori_cnt = cnt
            if not row[0] in ppid2index:
                ppid2index[row[0]] = cnt
                index2ppid[cnt] = row[0]
                cnt += 1
            if not row[1] in ppid2index:
                ppid2index[row[1]] = cnt
                index2ppid[cnt] = row[1]
                cnt += 1
            if [ppid2index[row[0]], ppid2index[row[1]]] == PPI[-1][:2]:
                PPI[-1] = [ppid2index[row[0]], ppid2index[row[1]], int(row[6])]
            else:
                PPI.append([ppid2index[row[0]], ppid2index[row[1]], int(row[6])])
            print(i)
        # if i > 10000:
        #     break
# print(f"{PPI}\n{ppid2index}\n{index2ppid}")
print(f"total count {cnt}")

#%% save PPI and make sparse matrix
num_protein = cnt
with open('ppid2index.json', 'w') as json_file:
  json.dump(ppid2index, json_file)
with open('index2ppid.json', 'w') as json_file:
  json.dump(index2ppid, json_file)
PPI = np.array(PPI, dtype=int)
np.savetxt('PPI.txt', PPI, fmt='%d')
PPI = coo_matrix((PPI[:,2], (PPI[:,0], PPI[:,1])), shape=(cnt, cnt), dtype=int)

#%% load PPI
data_type='full'
with open(f'/scratch/hdd001/home/haotian/Covid19Datasets/output/{data_type}/ppid2index.json', 'r') as json_file:
    ppid2index = json.load(json_file)
cnt = len(ppid2index)
PPI = np.loadtxt(f'/scratch/hdd001/home/haotian/Covid19Datasets/output/{data_type}/PPI.txt', dtype=int)
PPI = coo_matrix((PPI[:,2], (PPI[:,0], PPI[:,1])), shape=(cnt, cnt), dtype=int)
PPI = (PPI + PPI.T) / 2
PPI_numpy = PPI.todense()

# %% umap of PPI
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# %%
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(PPI.todense())
# embedding.shape
# # plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in iris.target])
# plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.4)
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the PPI', fontsize=24)
# plt.savefig('original.png')


# %% read names
name2ppid = {}
ppid2name = {}
with open('/scratch/hdd001/home/haotian/Covid19Datasets/PPI/9606.protein.info.v11.0.txt', 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    header = reader.__next__()
    for i, row in enumerate(reader):
        name2ppid[row[1]] = row[0]
        ppid2name[row[0]] = row[1]
# read viral related proteins
with open('/scratch/hdd001/home/haotian/Covid19Datasets/Cilicon/drugNetworks/drugbank_ranks.csv', 'r') as tsvfile:
    reader = csv.reader(tsvfile)
    header = reader.__next__()
proteins = []
proteins_id = []
for name in header:
    if 'HUMAN' in name:
        protein_name = name.split('_')[0]
        if protein_name in name2ppid:
            proteins.append(protein_name)
            proteins_id.append(ppid2index[name2ppid[protein_name]])

# %% convert ppid and uniprotid
ppid2uniprot = {}
uniprot2ppid = {}
with open('/scratch/hdd001/home/haotian/Covid19Datasets/output/ppid_uniprotid.tsv', 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    header = reader.__next__()
    print(header)
    ppid_col = header.index('Protein stable ID')
    uniprot_col = header.index(r"UniProtKB/Swiss-Prot ID")
    for i, row in enumerate(reader):
        ppid2uniprot['9606.'+row[ppid_col]] = row[uniprot_col]
        uniprot2ppid[row[uniprot_col]] = '9606.'+row[ppid_col]
        print(i)

# read drugbank data
# with open('/scratch/hdd001/home/haotian/Covid19Datasets/DrugBank/full database.xml') as fd:
#     drugbank = xmltodict.parse(fd.read())
# with open('/scratch/hdd001/home/haotian/Covid19Datasets/output/drugbank.json', 'w') as json_file:
#   json.dump(drugbank, json_file)
# drugbank['drugbank']['drug'][1]['targets']['target'][1]['polypeptide']['name']
# import xml.etree.ElementTree as ET
# tree = ET.parse('country_data.xml')
# root = tree.getroot()
target_drug_dict = {}
with open('/scratch/hdd001/home/haotian/Covid19Datasets/DrugBank/drugbank_all_targets.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = reader.__next__()
    col1 = header.index('UniProt ID')
    col2 = header.index('Uniprot Title')
    col3 = header.index('Drug IDs')
    species = header.index('Species')
    for i, row in enumerate(reader):
        if  row[species] == 'Humans':
            # FIXME: the id is kind of not aligned, fix here using the online upi conversion
            try:
                target_drug_dict[uniprot2ppid[row[col1]]] = row[col3].strip().split('; ')
            except: pass
        # if i > 6: break
drug_target_dict = {}
drug2index = {}
index2drug = {}
cnt = 0
for target, drugs in target_drug_dict.items():
    for drug in drugs:
        if not drug in drug_target_dict:
            drug_target_dict[drug] = [target]
            drug2index[drug] = cnt
            index2drug[cnt] = drug 
            cnt+=1
        else:
            drug_target_dict[drug].append(target)
drug_target_matrix = np.zeros([len(drug_target_dict),PPI_numpy.shape[1]])
for drug, targets in drug_target_dict.items():
    for target in targets:
        try:
            drug_target_matrix[drug2index[drug],ppid2index[target]] = 1.
        except: pass

# validation_drugs
drugs_in_trial = ['DB01117','DB01201','DB00608','DB00834','DB00431','DB09029','DB11574','DB09065','DB09054','DB09102',
                    'DB08880','DB11569','DB01058','DB00503','DB13179','DB01222','DB09212','DB00687','DB08865','DB09101']
for drug in drugs_in_trial: print(drug in drug2index)

# %% pca & diffusion
import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
# from knn import KNN
# from diffusion import Diffusion
from sklearn import preprocessing
from sklearn.decomposition import PCA
# from evaluate import compute_map_and_print
# %%
pca = PCA(n_components=50)
features = np.concatenate([PPI_numpy, drug_target_matrix], axis=0)
pca.fit(features)
PPI_pca = pca.transform(features)
PPI_pca = preprocessing.normalize(PPI_pca, norm="l2", axis=1)
print('finish computing pca')

n_proteins = PPI_numpy.shape[0]
reducer = umap.UMAP()
embedding = reducer.fit_transform(PPI_pca)
embedding.shape
plt.scatter(embedding[:n_proteins, 0], embedding[:n_proteins, 1], alpha=0.3)
plt.scatter(embedding[n_proteins:, 0], embedding[n_proteins:, 1], alpha=0.3)
for protein in ['ACE2','IDE','DDX10']:
    idx = ppid2index[name2ppid[protein]]
# ace2_id = ppid2index[name2ppid['ACE2']]
# ide_id = ppid2index[name2ppid['IDE']]
# ddx10_id = ppid2index[name2ppid['DDX10']]
    plt.scatter(embedding[idx, 0], embedding[idx, 1], alpha=0.3, c='red', s=60)
    plt.text(embedding[idx, 0], embedding[idx, 1], protein, c='red',
        horizontalalignment='left',
        verticalalignment='top')
for idx in proteins_id:
    plt.scatter(embedding[idx, 0], embedding[idx, 1], alpha=0.3, c='red', s=40)
plt.title('UMAP projection of the PPI \& drugs', fontsize=24)
plt.savefig('figure.png')

# %% diffusion here not showing better maps

# diffusion = Diffusion(preprocessing.normalize(PPI_numpy, norm="l2", axis=1), f'./.cache/{data_type}_cache')
# offline = diffusion.get_offline_results(1000, 60)
# features = preprocessing.normalize(offline, norm="l2", axis=1)

# pca = PCA(n_components=50)
# features = pca.fit_transform(features.todense())
# print('finish computing pca')
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(features)
# embedding.shape
# plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.4)
# plt.title('UMAP projection of the PPI', fontsize=24)
# plt.savefig('figure.png')



# %% Perform distance ranking
from sklearn.utils.graph_shortest_path import graph_shortest_path
num_drugs = len(drug_target_matrix)
whole_graph = np.hstack([np.vstack([PPI_numpy, drug_target_matrix]), np.vstack([drug_target_matrix.T, np.zeros([num_drugs, num_drugs])])]) > 0
whole_graph = whole_graph.astype(int)
from scipy import sparse
sparse_whole_graph = sparse.coo_matrix(whole_graph)
dist = graph_shortest_path(sparse_whole_graph) # (N,N)

sub_dist = dist[-num_drugs:][:, proteins_id]
drug_score = sub_dist.min(1)
np.argsort()

# Or 
idx = proteins_id + list(range(-num_drugs,0,1))
sub_graph = whole_graph[idx][:, idx]
sparse_sub_graph = sparse.coo_matrix(sub_graph)
dist = graph_shortest_path(sparse_sub_graph)
sub_dist = dist[-num_drugs:][:, -num_drugs]
drug_score = sub_dist.min(1)
rank = np.argsort(drug_score)
for id in rank[:20]: print(index2drug[id], index2drug[id] in drugs_in_trial)
