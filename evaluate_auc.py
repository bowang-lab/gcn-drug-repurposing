# given a setting, include:
# - what kind of embedding method
# - which types of data want to include
# provides the auc of the method
from sklearn.preprocessing import normalize
from multiscale_interactome.openne.node2vec import Node2vec
from multiscale_interactome.openne.graph import Graph
from multiscale_interactome.msi.node_to_node import DrugToIndication
import os
from multiscale_interactome.msi.msi import MSI
from multiscale_interactome.diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx
from sklearn.metrics import roc_auc_score

import pandas as pd
import urllib
import collections
import argparse
from parse_config import ConfigParser
from utils import query_uniprot2data, make_SARSCOV2_PPI


def graph_embedding(config, msi, gcn=False):
    # if node2vec
    # debug_g = nx.subgraph(msi.graph, list(g.nodes.keys())[:40])
    emb_file_prefix = config['node2vec']['eval_emb_file_prefix']
    walk_length = config['node2vec']['walk_length']
    number_walks = config['node2vec']['number_walk']
    emb_file = f"{emb_file_prefix}_num_{number_walks}_len_{walk_length}.embs.txt"
    gcn_emb_file = config['gcn']['emb_file']
    if not os.path.exists(emb_file):
        print("Calculating node2vec embeddings...")
        g = Graph()
        g.read_g(msi.graph)
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

    # FIXME: modify after this line
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

    return dp_saved


def main(config):
    method = config['method']
    save_graph_file = config['eval']['graph']
    diffusion_embs_dir = config['diffusion']['eval_diffusion_embs_dir']
    drug_to_indication = config['networks']['drug_to_indication']

    # Construct the multiscale interactome
    msi = MSI()
    msi.load()

    # store the whole graph
    if not os.path.exists(save_graph_file):
        nx.write_weighted_edgelist(msi.graph, save_graph_file)
    else:
        import warnings
        warnings.warn(
            f"graph struc file {save_graph_file} already exists. change this line if want to overwrite.")

    # nx.write_edgelist(msi.graph, 'whole_graph.edgelist')
    if method == 'diffusion':
        dp_saved = diffusion_method(diffusion_embs_dir, msi)
    elif method == 'node2vec':
        drugs_name_ranked = graph_embedding(config, msi, gcn=False)
    elif method == 'gcn' and config['gcn']['embs'] == "node2vec":
        drugs_name_ranked = graph_embedding(config, msi, gcn=True)
    else:
        raise NotImplementedError

    indication_graph = DrugToIndication(False, drug_to_indication)
    # list(indication_graph.graph['C0040038'])

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
    return msi


def evaluate():
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
    evaluate()
