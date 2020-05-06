# test proximity

from toolbox import wrappers
from utils import load_drugs_from, load_diseases_from

# load data
# network_file = "2016data/toy.sif"
# nodes_from = ["A", "C"]
# nodes_to = ["B", "D", "E"]
network_file = "2016data/network/networ.sif"
network = wrappers.get_network(file_name, only_lcc=True)

# provide drugs objects
drugs = load_drugs_from("2016data/target/drug_to_geneids.pcl.all")

# provide disease objects
disease = load_diseases_from("2016data/disease/disease_genes.tsv")

for x in drugs:
    for y in diseases:
        d, z, (mean, sd) = wrappers.calculate_proximity(
            network, x.nodes, y.nodes, min_bin_size=2, seed=452456)
