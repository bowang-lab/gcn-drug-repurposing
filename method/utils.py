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
    return response.decode('utf-8')


def make_SARSCOV2_PPI():
    ppi = pd.read_csv('data/viral_ppi/GordonEtAl-2020.tsv', sep='\t')
    name_list = ppi['Preys'].to_list()
    query = ' '.join(name_list).strip()
    string_id = query_uniprot2data(query=query).strip().split('\n')
    # ppi['string_id'] = string_id
    return string_id


def load_DTI():
    ppid2uniprot = {}
    uniprot2ppid = {}
    with open('data/ppid_uniprotid.tsv', 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        header = reader.__next__()
        print(header)
        ppid_col = header.index('Protein stable ID')
        uniprot_col = header.index(r"UniProtKB/Swiss-Prot ID")
        for i, row in enumerate(reader):
            ppid2uniprot['9606.'+row[ppid_col]] = row[uniprot_col]
            uniprot2ppid[row[uniprot_col]] = '9606.'+row[ppid_col]
            # print(i)
    target_drug_dict = {}
    with open('data/DrugBank/drugbank_all_targets.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = reader.__next__()
        col1 = header.index('UniProt ID')
        col2 = header.index('Uniprot Title')
        col3 = header.index('Drug IDs')
        species = header.index('Species')
        for i, row in enumerate(reader):
            if row[species] == 'Humans':
                # FIXME: the id is kind of not aligned, fix here using the online upi conversion
                try:
                    target_drug_dict[uniprot2ppid[row[col1]]
                                     ] = row[col3].strip().split('; ')
                except:
                    pass
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
                cnt += 1
            else:
                drug_target_dict[drug].append(target)
    return drug_target_dict


if __name__ == "__main__":
    # drugs = load_drugs_from("2016data/target/drug_to_geneids.pcl.all")
    # diseases = load_diseases_from("2016data/disease/disease_genes.tsv")
    # res = query_uniprot2data()
    pass
