# test proximity
import numpy as np
import pandas as pd

from toolbox import wrappers, network_utilities
from utils import load_drugs_from, load_diseases_from
import time

# load data
# network_file = "2016data/toy.sif"
# nodes_from = ["A", "C"]
# nodes_to = ["B", "D", "E"]
network_file = "2016data/network/network.sif"
network = wrappers.get_network(network_file, only_lcc=True)

# provide drugs objects
drugs = load_drugs_from("2016data/target/drug_to_geneids.pcl.all")

# provide disease objects
diseases = load_diseases_from("2016data/disease/disease_genes.tsv")

# The proximity has
# group group.name disease flag n.target n.disease n.overlap j.overlap d.target d.disease symptomatic ra re z.pathway z.side d z pval pval.adj
# where do we get all these info
# the drugs have group(DBID)
# dvd has the group, group.name disease, flag, n.target n.disease n.overlap

# star with loading the proximity
proximity = pd.read_csv('proximity.dat', sep=' ')

# update d and z
# (1.5915, 0.36657570841505577))
# mean, std = estimate_dist_stats_of()
# -- w/o computing mean, std
for i in range(3):  # (len(proximity)):
    start_time = time.time()
    row = proximity.loc[i]
    drug_name = row['group']
    disease_name = row['disease']
    d, z, (mean, sd) = wrappers.calculate_proximity(
        network, drugs[drug_name], diseases[disease_name])
    proximity.at[i, 'd'] = d
    proximity.at[i, 'z'] = z
    iter_time = time.time() - start_time
    print(f'iter {i}, {iter_time:.4f} s')

proximity.to_csv('result.dat', sep=' ')

# select = (proximity['disease']=) & ()
# proximity.loc[proximity.index[m & m2], 'z'] = 0

# for x in drugs[:3]:
#     for y in diseases[:3]:
#         d, z, (mean, sd) = wrappers.calculate_proximity(
#             network, x.nodes, y.nodes, min_bin_size=2, seed=452456)
#         print(d, z, (mean, sd))
