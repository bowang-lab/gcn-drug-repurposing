from utils import convert_name_list
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
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


# -------------------------------------------------
msi = MSI()
msi.load()

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

dp_saved.drug_or_indication2diffusion_profile["DB00642"]


class DrugToIndication():
    def __init__(self, directed, file_path, sep="\t"):
        self.file_path = file_path
        self.directed = directed
        self.sep = sep
        self.load()

    def load_df(self):
        df = pd.read_csv(self.file_path, sep=self.sep,
                         index_col=False, dtype=str)
        self.df = df

    def load_edge_list(self):
        # Creates directional edgelist from node_1 to node_2
        assert(not(self.df is None))
        edge_list = list(zip(self.df["drug"], self.df["indication"]))
        self.edge_list = edge_list

    def load_graph(self):
        if (self.directed):
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        self.graph.add_edges_from(self.edge_list)

    def update_node2attr(self, node2type, col_1, col_2):
        for node, type_ in zip(self.df[col_1], self.df[col_2]):
            if node in node2type:
                assert((node2type[node] == type_) or (
                    pd.isnull(node2type[node]) and pd.isnull(type_)))
            else:
                node2type[node] = type_
        return node2type

    def load_node2type(self):
        assert(not(self.df is None))
        node2type = dict()
        node2type = self.update_node2attr(node2type, "node_1", "node_1_type")
        node2type = self.update_node2attr(node2type, "node_2", "node_2_type")
        self.node2type = node2type

    def load_type2nodes(self):
        type2nodes = dict()
        for node, type_ in self.node2type.items():
            if type_ in type2nodes:
                type2nodes[type_].add(node)
            else:
                type2nodes[type_] = set([node])
        self.type2nodes = type2nodes

    def load_node2name(self):
        assert(not(self.df is None))
        node2name = dict()
        node2name = self.update_node2attr(node2name, "drug", "drug_name")
        node2name = self.update_node2attr(
            node2name, "indication", "indication_name")
        self.node2name = node2name

    def load_name2node(self):
        assert(not(self.df is None))
        name2node = {v: k for k, v in self.node2name.items()}
        self.name2node = name2node

    def load(self):
        assert(not(self.file_path is None))
        self.load_df()
        self.load_edge_list()
        self.load_graph()
        self.load_node2name()
        self.load_name2node()


indication_graph = DrugToIndication(False, 'data/drug_indication_df.tsv')
list(indication_graph.graph['C0040038'])

drugs_index_in_msi = []
drugs = []
indications_index_in_msi = []
indications = []
for i, node in enumerate(msi.nodelist):
    if msi.graph.nodes[node]['type'] == 'drug':
        drugs.append(node)
        drugs_index_in_msi.append(i)
    if msi.graph.nodes[node]['type'] == 'indication':
        indications.append(node)
        indications_index_in_msi.append(i)
# list of indications

num_drugs = len(drugs_index_in_msi)
all_aucs = []
for indication in indications:
    # build ref
    ref = np.zeros(num_drugs, dtype=int)
    for pos_drug in list(indication_graph.graph[indication]):
        ref[drugs.index(pos_drug)] = 1
    # predict vector
    tmp = dp_saved.drug_or_indication2diffusion_profile[indication]
    predict = tmp[drugs_index_in_msi]
    auc = roc_auc_score(ref, predict)
    # print(auc)
    all_aucs.append(auc)
all_aucs = np.array(all_aucs)
print(f"median auc: {np.median(all_aucs)}, mean auc: {all_aucs.mean()}")

# -------------------------------------------------
drug = 'DB00747'
degree = len(msi.graph[drug])
degree_list = [len(msi.graph[drug]) for drug in drugs]
plt.figure()
plt.hist(degree_list, bins=[0, 1, 2, 4, 16, 64, 256])

plt.savefig('degree.png')

# -------------------------------------------------
permutations = pd.read_csv(
    'data/StringDatabaseWithLogFcOfGenesPassing5pFDRcutoff.tsv', sep='\t')
# permutations = permutations[permutations['ChangesAt24hours'].notna(
# ) | permutations['ChangesDueToCovidAt24Hour'].notna()]
permutations = permutations[(permutations['ChangesDueToCovidAt24Hour'] > 2) | (
    permutations['ChangesDueToCovidAt24Hour'] < -2)]
permutations['avg'] = permutations[['ChangesAt24hours',
                                    'ChangesDueToCovidCovariate', 'ChangesDueToCovidAt24Hour']].mean(axis=1)
permutations = permutations.drop(columns=['annotation', 'protein_size'])

perm_StringID = list(permutations['protein_external_id'])
perm_UniprotKB_ID = convert_name_list(
    perm_StringID, from_data='STRING_ID', to_data='ID')
perm_protein_name = convert_name_list(
    perm_UniprotKB_ID, from_data='ID', to_data='GENENAME')

# -------------------------------------------------
# cyclica_DTI = pd.read_csv('bioactivities_bioactivities.csv')
# cyclica_DTI.columns.values
# array(['Unnamed: 0', 'accession', 'article_doi', 'entry_name',
#        'isomeric_smiles', 'op', 'orig_unit', 'orig_value',
#        'patent_number', 'pubchem_aid', 'pubmed', 'source_db',
#        'source_link', 'std_unit', 'std_value', 'stitch_confidence_score',
#        'type'], dtype=object)

# perm_pathways = pd.read_csv(
#     'data/04_Immune_Genes_Enrichment_GO_MF_BP_Intersection.tsv', sep='\t')
# pathway_ID = list(set(perm_pathways['Pathway_ID']))


# -------------------------------------------------
gcn_embs = np.loadtxt("whole_graph_gcn_pathway.embs.txt")
gcn_embs = normalize(gcn_embs, axis=1)
ppi_embs = np.loadtxt(
    "whole_graph_node2vec_pathway_walk_num_64_len_16.embs.txt", skiprows=1, dtype=object)
nodes = list(ppi_embs[:, 0])
del ppi_embs
node_names = [msi.node2name[node] for node in nodes]

covid_emb = gcn_embs[nodes.index('NodeCovid')]
protein_mask = [True if (msi.graph.node[node]['type'] ==
                         'protein' and msi.node2name[node] is not np.nan) else False for node in nodes]
protein_names = list(np.array(node_names)[protein_mask])
protein_embs = gcn_embs[protein_mask]
protein_proximities = list(np.matmul(protein_embs, covid_emb))


shortest_paths = []
path_lengths = []
for node in list(np.array(nodes)[protein_mask]):
    # compute paths
    path = nx.shortest_path(
        msi.graph, source=node, target='NodeCovid')
    path_node_names = [node if msi.node2name[node]
                       is np.nan else msi.node2name[node] for node in path]
    shortest_paths.append(', '.join(path_node_names))
    path_lengths.append(len(path)-1)


protein_df = pd.DataFrame({
    'protein name': protein_names,
    'proximity to covid-19 node': protein_proximities,
    'shortest path to Covid': shortest_paths,
    'path length': path_lengths
})
protein_df.to_csv('all_protein_proximities_pathway_gcn.tsv',
                  sep='\t', na_rep='NA', index=False)
