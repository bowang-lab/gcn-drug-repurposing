from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles

import pandas as pd
import urllib


def query_uniprot2data(query='P40925 P40926', to_data='String', style='list'):
    if to_data == 'String':
        target = 'STRING_ID'
    else:
        target = to_data
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


def make_SARSCOV2_PPI(to_data='GENENAME'):
    ppi = pd.read_csv('data/viral_ppi/GordonEtAl-2020.tsv', sep='\t')
    name_list = ppi['Preys'].to_list()
    query = ' '.join(name_list).strip()
    string_id = query_uniprot2data(
        query=query, to_data=to_data).strip().split('\n')
    # ppi['string_id'] = string_id
    return string_id


covid_protein_list = make_SARSCOV2_PPI()  # 336
# has 302
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
covid_protein_df.to_csv('data/covid_to_protein.tsv', sep='\t', index=False)

# Construct the multiscale interactome
msi = MSI(indication2protein_file_path="data/covid_to_protein.tsv",
          indication2protein_directed=False)
msi.load()


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
    save_load_file_path="results/covid/"
)
dp.calculate_diffusion_profiles(msi)

# Load saved diffusion profiles
dp_saved = DiffusionProfiles(
    alpha=None,
    max_iter=None,
    tol=None,
    weights=None,
    num_cores=None,
    save_load_file_path="results/covid/"
)
msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

dp_saved.drug_or_indication2diffusion_profile["DB01098"]
dp_saved.drug_or_indication2diffusion_profile["NodeCovid"]

res = dp_saved.drug_or_indication2diffusion_profile["NodeCovid"]
drugs = []
proximities = []
assert len(res) == len(msi.nodelist)
for i, node in enumerate(msi.nodelist):
    if msi.graph.nodes[node]['type'] == 'drug':
        drugs.append(node)
        proximities.append(res[i])

proximities_ranked_id = np.argsort(np.array(proximities))[::-1]
drugs_ranked = [drugs[i] for i in proximities_ranked_id]
drugs_name_ranked = [msi.node2name[drug] for drug in drugs_ranked]
