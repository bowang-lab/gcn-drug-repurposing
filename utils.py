import urllib.request
import urllib.parse
from modules.netmodules import DrugModule, DiseaseModule
import pickle
import csv
import numpy as np
import pandas as pd


def load_drugs_from(file):
    """the file should be a pkl binary file."""

    drug_data = pickle.load(open(file, 'rb'))
    # the drug_data is a dict like the following:
    #  'DB01113': {'10846', '5142'},
    #  'DB01114': {'3269', '6530', '6531', '6532'},
    #  'DB01115': {'3736', '775', '776', '779', '781', '783', '801', '8912'},

    drugs = {}
    for drug_id, genes in drug_data.items():
        drug = DrugModule(name=drug_id, connected_nodes=genes)
        drugs.update(drug.to_dict())
    return drugs


def load_diseases_from(file):
    """reads disease connections from tsv data file"""
    with open(file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        disease_names = []
        connections = []
        for i, row in enumerate(reader):
            # row[0] is empty
            disease_names.append(row[1])
            connections.append(row[2:])

    diseases = {}
    for i, name in enumerate(disease_names):
        entry = DiseaseModule(
            name=name, connected_nodes=connections[i])
        diseases.update(entry.to_dict())
    return diseases


def query_uniprot2data(query='P40925 P40926', to_data='String', style='list'):
    if to_data == 'String':
        target = 'STRING_ID'
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
    return response


def make_SARSCOV2_PPI():
    ppi = pd.read_csv('data/viral_ppi/GordonEtAl-2020.tsv', sep='\t')
    name_list = ppi['Preys'].to_list()
    query = ' '.join(name_list).strip()
    string_id = query_uniprot2data(query=query).decode(
        'utf-8').strip().split('\n')
    # ppi['string_id'] = string_id
    return string_id


if __name__ == "__main__":
    # drugs = load_drugs_from("2016data/target/drug_to_geneids.pcl.all")
    # diseases = load_diseases_from("2016data/disease/disease_genes.tsv")
    res = query_uniprot2data()
