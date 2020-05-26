from sklearn.preprocessing import normalize
from openne.node2vec import Node2vec
from openne.graph import Graph
import os
from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles

import pandas as pd
import urllib


def query_uniprot2data(query='P40925 P40926', to_data='String', style='list'):
    if to_data == 'String':
        target = 'STRING_ID'
    else:
        target = to_data
    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': 'ACC+ID',
        'to': target,
        'format': style,
        'query': query
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
    # print(response.decode('utf-8'))
    return response.decode('utf-8')


def make_SARSCOV2_PPI(to_data='GENENAME'):
    ppi = pd.read_csv(
        '/h/haotian/Code/deep-drug-repurposing/data/viral_ppi/GordonEtAl-2020.tsv', sep='\t')
    name_list = ppi['Preys'].to_list()
    query = ' '.join(name_list).strip()
    string_id = query_uniprot2data(
        query=query, to_data=to_data).strip().split('\n')
    # ppi['string_id'] = string_id
    return string_id


covid_to_protein = 'data/covid_to_protein.tsv'
if not os.path.exists(covid_to_protein):
    # load proteins
    proteins = pd.read_csv('data/protein_to_protein.tsv', sep='\t')

    covid_protein_list = make_SARSCOV2_PPI()  # 332
    # has 306
    covid_protein_list = [
        protein for protein in covid_protein_list if protein in proteins['node_1_name'].values]

    proteinname2node = dict(
        set(list(zip(proteins['node_1_name'], proteins['node_1']))))
    # make covid data fram
    node_1 = ['NodeCovid'] * len(covid_protein_list)
    node_1_type = ['indication'] * len(covid_protein_list)
    node_1_name = ['covid-19'] * len(covid_protein_list)
    node_2 = [proteinname2node[name] for name in covid_protein_list]
    node_2_type = ['protein'] * len(covid_protein_list)
    node_2_name = covid_protein_list
    covid_protein_df = pd.DataFrame({
        'node_1': node_1,
        'node_2': node_2,
        'node_1_type': node_1_type,
        'node_2_type': node_2_type,
        'node_1_name': node_1_name,
        'node_2_name': node_2_name
    })
    covid_protein_df.to_csv(covid_to_protein, sep='\t', index=False)
else:
    covid_protein_df = pd.read_csv(covid_to_protein, sep='\t')

# Construct the multiscale interactome
msi = MSI(indication2protein_file_path=covid_to_protein,
          indication2protein_directed=False)
msi.load()


diffusion_embs_dir = "results/covid/"
if not os.path.exists(diffusion_embs_dir):
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
        save_load_file_path="results/covid/"
    )
    dp.calculate_diffusion_profiles(msi)

# Load saved diffusion profiles
dp_saved = DiffusionProfiles(
    alpha=None,
    max_iter=None,
    tol=None,
    weights=None,
    num_cores=None,
    save_load_file_path="results/covid/"
)
msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

dp_saved.drug_or_indication2diffusion_profile["NodeCovid"]

res = dp_saved.drug_or_indication2diffusion_profile["NodeCovid"]
drugs = []
proximities = []
assert len(res) == len(msi.nodelist)
for i, node in enumerate(msi.nodelist):
    if msi.graph.nodes[node]['type'] == 'drug':
        drugs.append(node)
        proximities.append(res[i])

proximities_ranked_id = np.argsort(np.array(proximities))[::-1]
drugs_ranked = [drugs[i] for i in proximities_ranked_id]
drugs_name_ranked = [drug if msi.node2name[drug]
                     is np.nan else msi.node2name[drug] for drug in drugs_ranked]

with open('drugs_candidataes.txt', 'w') as f:
    f.write('\n'.join(drugs_name_ranked[:40]))

# store the whole graph
if not os.path.exists('whole_graph.weighted.edgelist'):
    nx.write_weighted_edgelist(msi.graph, 'whole_graph.weighted.edgelist')
    # nx.write_edgelist(msi.graph, 'whole_graph.edgelist')


# if node2vec
# debug_g = nx.subgraph(msi.graph, list(g.nodes.keys())[:40])
walk_length = 16
number_walks = 64
emb_file = f"whole_graph_node2vec_walk_num_{number_walks}_len_{walk_length}.embs.txt"
if not os.path.exists(emb_file):
    g = Graph()
    g.read_g(msi.graph)
    model = Node2vec(
        graph=g, path_length=walk_length,
        num_paths=64, dim=128,
        workers=8, p=0.25, q=0.25, window=10
    )
    print("Saving embeddings...")
    model.save_embeddings(emb_file)


# load embs
node_vecs = np.loadtxt(emb_file, skiprows=1, dtype=object)
node_names = list(node_vecs[:, 0])
if gcn:
    node_embs = np.loadtxt('whole_graph_gcn.embs.txt')
    node_embs = normalize(node_embs, axis=1)
else:
    node_embs = node_vecs[:, 1:].astype(np.float)


covid_emb = np.array(node_embs[node_names.index('NodeCovid')])

drug_names = []
drug_embs = []
for i, node in enumerate(node_names):
    if msi.graph.nodes[node]['type'] == 'drug':
        drug_names.append(node)
        drug_embs.append(node_embs[i])
drug_embs = np.array(drug_embs)
proximities = np.matmul(drug_embs, covid_emb)
# drug_embs_normed = normalize(drug_embs, axis=1)
# proximities = np.matmul(drug_embs_normed, covid_emb)

proximities_ranked_id = np.argsort(np.array(proximities))[::-1]
drugs_ranked = [drug_names[i] for i in proximities_ranked_id]
drugs_name_ranked = [drug if msi.node2name[drug]
                     is np.nan else msi.node2name[drug] for drug in drugs_ranked]


with open('drugs_candidataes.txt', 'w') as f:
    f.write('\n'.join(drugs_name_ranked[:40]))
