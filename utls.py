from modules.netmodules import DrugModule
import pickle


def load_drugs_from(file):
    """the file should be a pkl binary file."""

    drug_data = pickle.load(open(file, 'rb'))
    # the drug_data is a dict like the following:
    #  'DB01113': {'10846', '5142'},
    #  'DB01114': {'3269', '6530', '6531', '6532'},
    #  'DB01115': {'3736', '775', '776', '779', '781', '783', '801', '8912'},

    drugs = []
    for drug_id, genes in drug_data.items():
        drugs.append(DrugModule(name=drug_id, connected_nodes=genes))
    return drugs


def load_diseases_from():
    """disease """
    return


if __name__ == "__main__":
    load_drugs_from("2016data/target/drug_to_geneids.pcl.all")
