from toolbox import wrappers
file_name = "data/toy.sif"
network = wrappers.get_network(file_name, only_lcc = True)
nodes_from = ["A", "C"]
nodes_to = ["B", "D", "E"]
d, z, (mean, sd) = wrappers.calculate_proximity(network, nodes_from, nodes_to, min_bin_size = 2, seed=452456)
print (d, z, (mean, sd))