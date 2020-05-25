# given a setting, include:
# - what kind of embedding method
# - which types of data want to include
# provides an interface that you provide a disease and its related proteins,
# generates a ranked list of proposed drugs.
import collections
import argparse
from parse_config import ConfigParser


def main(config):
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Drug Repurposing')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-s', '--save-dir', default=None, type=str,
                      help='path to save and load (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        # CustomArgs(['--lr', '--learning_rate'], type=float,
        #            target=('optimizer', 'args', 'lr')),
        # CustomArgs(['--bs', '--batch_size'], type=int,
        #            target=('data_loader', 'args', 'batch_size')),
        # CustomArgs(['--name'], type=str, target=('name', )),
        # CustomArgs(['--dataset_type'], type=str, target=('dataset', 'type')),
        # CustomArgs(['--data_name'], type=str,
        #            target=('dataset', 'args', 'data_name')),
        # CustomArgs(['--n_clusers'], type=int,
        #            target=('dataset', 'args', 'n_clusers')),
        # CustomArgs(['--topk'], type=int, target=('dataset', 'args', 'topk')),
        # CustomArgs(['--epochs'], type=int, target=('trainer', 'epochs')),
        # CustomArgs(['--layers'], type=str, target=('arch', 'args', 'layers')),
    ]
    config = ConfigParser(args, options)
    main(config)
