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


if __name__ == "__main__":
    drugs = load_drugs_from("2016data/target/drug_to_geneids.pcl.all")
    diseases = load_diseases_from("2016data/disease/disease_genes.tsv")
