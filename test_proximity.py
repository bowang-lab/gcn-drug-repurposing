# test proximity
import multiprocessing
from joblib import Parallel, delayed
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
# bins = network_utilities.get_degree_binning(network, bin_size=100)
# for i in range(len(proximity)):
#     start_time = time.time()
#     row = proximity.loc[i]
#     drug_name = row['group']
#     disease_name = row['disease']
#     d, z, (mean, sd) = wrappers.calculate_proximity(
#         network, drugs[drug_name], diseases[disease_name], bins=bins)
#     proximity.at[i, 'd'] = d
#     proximity.at[i, 'z'] = z
#     iter_time = time.time() - start_time
#     print(f'iter {i}, {iter_time:.4f} s')

# proximity.to_csv('result.dat', sep=' ', na_rep='NA')

# ==================================================
# -- parallel version
inputs = range(len(proximity))
bins = network_utilities.get_degree_binning(network, bin_size=100)


def processInput(i):
    try:
        start_time = time.time()
        row = proximity.loc[i]
        drug_name = row['group']
        disease_name = row['disease']
        d, z, (mean, sd) = wrappers.calculate_proximity(
            network, drugs[drug_name], diseases[disease_name], bins=bins)
        iter_time = time.time() - start_time
        print(f'iter {i}, {iter_time:.4f} s')
        return (i, d, z)
    except:
        print(f"iter {i} is missing")


num_cores = 16  # multiprocessing.cpu_count()
print(f"parallel on {num_cores} cores")

total_time = time.time()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
total_time = time.time() - total_time
print(f'total time: {total_time} s')

print(results)
for i, d, z in results:
    proximity.at[i, 'd'] = d
    proximity.at[i, 'z'] = z
proximity.to_csv('result.dat', sep=' ', na_rep='NA')
# ==================================================


# select = (proximity['disease']=) & ()
# proximity.loc[proximity.index[m & m2], 'z'] = 0

# for x in drugs[:3]:
#     for y in diseases[:3]:
#         d, z, (mean, sd) = wrappers.calculate_proximity(
#             network, x.nodes, y.nodes, min_bin_size=2, seed=452456)
#         print(d, z, (mean, sd))
