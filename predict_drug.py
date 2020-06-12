# given a setting, include:
# - what kind of embedding method
# - which types of data want to include
# provides an interface that you provide a disease and its related proteins,
# generates a ranked list of proposed drugs.
from sklearn.preprocessing import normalize
from multiscale_interactome.openne.node2vec import Node2vec
from multiscale_interactome.openne.graph import Graph
import os
from multiscale_interactome.msi.msi import MSI
from multiscale_interactome.diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

import pandas as pd
import urllib
import collections
import argparse
from parse_config import ConfigParser
from utils import query_uniprot2data, make_SARSCOV2_PPI
from utils import convert_name_list


def graph_embedding(config, msi, gcn=False):
    # if node2vec
    # debug_g = nx.subgraph(msi.graph, list(g.nodes.keys())[:40])
    emb_file_prefix = config['node2vec']['emb_file_prefix']
    walk_length = config['node2vec']['walk_length']
    number_walks = config['node2vec']['number_walk']
    emb_file = f"{emb_file_prefix}_num_{number_walks}_len_{walk_length}.embs.txt"
    gcn_emb_file = config['gcn']['emb_file']
    if not os.path.exists(emb_file):
        g = Graph()
        g.read_g(msi.graph)
        # print('assigning initail weights to 1.0')
        # for i, j in g.G.edges():
        #     g.G[i][j]['weight'] = 1.0
        model = Node2vec(
            graph=g, path_length=walk_length,
            num_paths=number_walks, dim=128,
            workers=8, p=0.25, q=0.25, window=10
        )
        print("Saving embeddings...")
        model.save_embeddings(emb_file)

    # load embs
    node_vecs = np.loadtxt(emb_file, skiprows=1, dtype=object)
    node_names = list(node_vecs[:, 0])
    if gcn:
        node_embs = np.loadtxt(gcn_emb_file)
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
    return drugs_name_ranked


def diffusion_method(diffusion_embs_dir, msi):
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
            save_load_file_path=diffusion_embs_dir
        )
        dp.calculate_diffusion_profiles(msi)
    # Load saved diffusion profiles
    dp_saved = DiffusionProfiles(
        alpha=None,
        max_iter=None,
        tol=None,
        weights=None,
        num_cores=None,
        save_load_file_path=diffusion_embs_dir
    )
    msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
    dp_saved.load_diffusion_profiles(
        msi.drugs_in_graph + msi.indications_in_graph)

    # provide drug candidates
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

    return drugs_name_ranked


def main(config):
    method = config['method']
    save_drug_candidates_dir = config['output']['drug_candidates']
    save_graph_file = config['output']['graph']
    topk = config['topk']
    gordon_viral_protein = config['networks']['gordon_viral_protein']
    covid_to_protein = config['covid']['save_dir']
    protein_to_protein = config['networks']['protein_to_protein']
    diffusion_embs_dir = config['diffusion']['diffusion_embs_dir']

    if not os.path.exists(covid_to_protein):
        # load proteins
        proteins = pd.read_csv(protein_to_protein, sep='\t')

        covid_protein_list = make_SARSCOV2_PPI(gordon_viral_protein)  # 332
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
    print('assigning weights')
    weights = {
        'down_functional_pathway': 4.4863053901688685,
        'indication': 3.541889556309463,
        'functional_pathway': 6.583155399238509,
        'up_functional_pathway': 2.09685000906964,
        'protein': 4.396695660380823,
        'drug': 3.2071696595616364
    }
    msi.weight_graph(weights)
    if config['covid'].get('add_pathway'):
        print('adding pathway connections')
        perm_pathways = pd.read_csv(
            config['covid']['pertub_pathway_file'], sep='\t')
        pathway_ID = list(set(perm_pathways['Pathway_ID']))
        for pathway in pathway_ID:
            if pathway in msi.graph.nodes:
                try:
                    msi.graph.add_edge('NodeCovid', pathway,
                                       weight=3.0/len(pathway_ID))
                    msi.graph.add_edge(pathway, 'NodeCovid',
                                       weight=3.0/len(pathway_ID))
                    print(f"find {pathway}")
                except Exception:
                    print(Exception)
    if config['covid']['add_permutation']:
        print('adding pertubation protein connections')
        permutations = pd.read_csv(
            config['covid']['permutation_file'], sep='\t')
        # permutations = permutations[permutations['ChangesAt24hours'].notna(
        # ) | permutations['ChangesDueToCovidAt24Hour'].notna()]
        permutations = permutations[(permutations['ChangesDueToCovidAt24Hour'] > 2) | (
            permutations['ChangesDueToCovidAt24Hour'] < -2)]
        permutations['avg'] = permutations[['ChangesAt24hours',
                                            'ChangesDueToCovidCovariate', 'ChangesDueToCovidAt24Hour']].mean(axis=1)
        permutations = permutations.drop(
            columns=['annotation', 'protein_size'])

        perm_StringID = list(permutations['protein_external_id'])
        perm_UniprotKB_ID = convert_name_list(
            perm_StringID, from_data='STRING_ID', to_data='ID')
        perm_protein_name = convert_name_list(
            perm_UniprotKB_ID, from_data='ID', to_data='GENENAME')
        for protein in perm_protein_name:
            try:
                protein_node = msi.name2node[protein]
                msi.graph.add_edge('NodeCovid', protein_node)
                msi.graph.add_edge(protein_node, 'NodeCovid')
            except Exception:
                print(Exception)
        print(msi.graph['10349'])

    # store the whole graph
    if not os.path.exists(save_graph_file):
        nx.write_weighted_edgelist(msi.graph, save_graph_file)
    else:
        import warnings
        warnings.warn(
            f"graph struc file {save_graph_file} already exists. change this line if want to overwrite.")

    # nx.write_edgelist(msi.graph, 'whole_graph.edgelist')
    if method == 'diffusion':
        drugs_name_ranked = diffusion_method(diffusion_embs_dir, msi)
    elif method == 'node2vec':
        drugs_name_ranked = graph_embedding(config, msi, gcn=False)
    elif method == 'gcn' and config['gcn']['embs'] == "node2vec":
        drugs_name_ranked = graph_embedding(config, msi, gcn=True)
    else:
        raise NotImplementedError
    with open(save_drug_candidates_dir, 'w') as f:
        f.write('\n'.join(drugs_name_ranked[:topk]))
    return msi


def run_predict():
    args = argparse.ArgumentParser(description='Drug Repurposing')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-s', '--save-dir', default=None, type=str,
                      help='path to save and load (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        # CustomArgs(['--lr', '--learning_rate'], type=float,
        #            target=('optimizer', 'args', 'lr')),
        # CustomArgs(['--bs', '--batch_size'], type=int,
        #            target=('data_loader', 'args', 'batch_size')),
        # CustomArgs(['--name'], type=str, target=('name', )),
        # CustomArgs(['--dataset_type'], type=str, target=('dataset', 'type')),
        # CustomArgs(['--data_name'], type=str,
        #            target=('dataset', 'args', 'data_name')),
        # CustomArgs(['--n_clusers'], type=int,
        #            target=('dataset', 'args', 'n_clusers')),
        # CustomArgs(['--topk'], type=int, target=('dataset', 'args', 'topk')),
        # CustomArgs(['--epochs'], type=int, target=('trainer', 'epochs')),
        # CustomArgs(['--layers'], type=str, target=('arch', 'args', 'layers')),
    ]
    config = ConfigParser(args, options)
    return main(config)


if __name__ == '__main__':
    run_predict()
