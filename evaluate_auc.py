# given a setting, include:
# - what kind of embedding method
# - which types of data want to include
# provides the auc of the method
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
