# Drug Repurposing

## Installation

1. `pip install -r requirements.txt`
2. follow the instruction [here](https://github.com/thunlp/OpenNE) to install openne. For the specific version we used, check the submodule OpenNE in the resources folder.

We inherited and edited the data and network buildup from [MSI](https://github.com/snap-stanford/multiscale-interactome). Credits to the original authors. Our edited version is stored in folder [multiscale](multiscale).

## Predict the proximities for all target proteins and drugs with respect to Covid-19

The predictions will be stored in a `tsv` file.
```bash
python run_covid.py
```

## Predict drug candidates for covid

- Download the data [here](https://drive.google.com/drive/folders/1W9G2Zxq385FlJSWaB3-wxsmBXTpfrPl2?usp=sharing) and set the data directories in `config.json`.

- Use `python predict_drug.py -c config.json` to generate a drug list and store in a output file assigned in `config.json`.

It is recommended to put all settings in a config file, i.e. `config.json`.

## Use the biological graph object (msi)

```python
from predict_drug import run_predict
msi = run_predict()

# retrieve the nodes in graph
msi.graph.nodes

# check the connections of a node
msi.graph['DB04865']
```

Currently we are using [networkx](https://networkx.github.io/documentation/stable/). Click and find more interfaces from the documents.
The similarity loss GCN model used is inspired from Guided Similarity Separation([GSS](https://github.com/layer6ai-labs/GSS))
