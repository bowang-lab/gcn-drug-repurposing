from __future__ import division

import argparse
import os
import torch
import numpy as np
from modules.model import *
from method.dataset import DiffusionDataLoader, DiffusionDataSet

from helpers.helper import *

parser = argparse.ArgumentParser()
parser.add_argument('--kq', type=int, default=5,
                    help='Top k number for the query graph.')
parser.add_argument('--k', type=int, default=5,
                    help='Top k number for the index graph.')
parser.add_argument('--alpha', type=float, default=1,
                    help='Parameter alpha for gss loss.')
parser.add_argument('--beta', type=float, default=None,
                    help='Parameter beta for gss loss.')
parser.add_argument('--beta-percentile', type=float, default=None,
                    help='Automatically select beta by the percentile of similarity matrix''s distribution.')
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=0,
                    help='Batch size. Set to 0 (default) if train the batch of all samples.')
parser.add_argument('--hidden-units', type=int, default=128,
                    help='Number of units in hidden layer')
parser.add_argument('--num-layers', type=int,
                    default=2, help='Number of layers')
parser.add_argument('--loss', type=str, default='gss',
                    help='Set the loss type.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--init-weights', type=float, default=1e-5,
                    help='The std of the off-diagonal elements of the weights in GCN referred as epsilon.')
parser.add_argument('--regularizer-scale', type=float,
                    default=1e-5, help='The scale of l2 regularization')
parser.add_argument('--layer-decay', type=float, default=0.3,
                    help='Residual GCN layer decay.')
parser.add_argument('--dataset', type=str, default='roxford5k', choices=['roxford5k', 'rparis6k', 'instre'],
                    help='Dataset.')
parser.add_argument('--emb-file', type=str, default=None,
                    help='embedding file name.')
parser.add_argument('--data-path', type=str, default=None,
                    help='Dataset files location.')
parser.add_argument('--gpu-id', type=int, default=None,
                    help='Which GPU to use. By default None means training on CPU.')
parser.add_argument("--report-hard", help="If evaluate on Hard setup or Medium. It doesn't matter to INSTRE",
                    action="store_true")
parser.add_argument("--graph-mode", type=str, default='descriptor',
                    choices=['descriptor', 'ransac', 'approx_ransac'],
                    help="Choose the way to construct kNN graph. Descriptor mode uses the "
                         "inner product of dense descriptors referred as GSS in the GeM+GSS. Ransac "
                         "mode applies spatial verification on both query and index graphs referred as "
                         "GeM+GSS_V-SV. Approx_ransac mode is a fast inference method where spatial "
                         "verification is only applied on index graph during offline training phase "
                         "referred as GeM+GSS_V.")
args = parser.parse_args()
for key in vars(args):
    print(key + ":" + str(vars(args)[key]))

if args.beta is not None and args.beta_percentile is not None:
    raise Exception(
        'beta and beta_percentile can not be used at the same time!')


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Q, X = load_data(args.dataset, args.data_path)
    ppi_embs = np.loadtxt(args.emb_file, skiprows=1, dtype=object)
    ppi_embs = ppi_embs[:, 1:].astype(np.float)
    X = ppi_embs.T  # (d, num_nodes)
    Q = X[:, 0:3]

    if args.graph_mode == 'ransac':
        q_RANSAC_graph, x_RANSAC_graph = load_ransac_graph(
            args.dataset, args.data_path)
    elif args.graph_mode == 'approx_ransac':
        _, x_RANSAC_graph = load_ransac_graph(args.dataset, args.data_path)
        q_RANSAC_graph = None
    else:
        q_RANSAC_graph, x_RANSAC_graph = None, None

    q_adj, q_features, x_adj, x_features = gen_graph(
        Q, X, args.kq, args.k, q_RANSAC_graph, x_RANSAC_graph)

    all_features = np.concatenate([q_features, x_features])
    all_adj = combine_graph(q_adj, x_adj)

    all_adj_normed = preprocess_graph(all_adj)
    x_adj_normed = preprocess_graph(x_adj)
    x_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(
        x_adj_normed)
    all_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(
        all_adj_normed)

    # features_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, args.hidden_units])
    # adj_placeholder = tf.sparse_placeholder(dtype=tf.float32, shape=[None, None])

    # FIXME: regularizer = tf.contrib.layers.l2_regularizer(scale=args.regularizer_scale)

    model = ResidualGraphConvolutionalNetwork(train_batch_size=args.batch_size if args.batch_size > 0 else x_adj_normed.shape[0],
                                              val_batch_size=all_adj_normed.shape[0],
                                              num_layers=args.num_layers,
                                              hidden_units=args.hidden_units,
                                              init_weights=args.init_weights,
                                              layer_decay=args.layer_decay)

    if args.gpu_id is not None:
        model.cuda()
        x_adj_normed_sparse_tensor = x_adj_normed_sparse_tensor.cuda()
        all_adj_normed_sparse_tensor = all_adj_normed_sparse_tensor.cuda()
        print('using gpu')

    # FIXME: add flexible iterator
    training_dataset = DiffusionDataSet(features=x_features,
                                        adj=x_adj_normed_sparse_tensor)
    training_loader = DiffusionDataLoader(training_dataset,
                                          batch_size=args.batch_size if args.batch_size > 0 else len(
                                              training_dataset),
                                          num_workers=6,
                                          shuffle=True)
    validation_dataset = DiffusionDataSet(features=all_features,
                                          adj=all_adj_normed_sparse_tensor)
    validation_loader = DiffusionDataLoader(validation_dataset,
                                            batch_size=len(validation_dataset),
                                            num_workers=6,
                                            shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=0)

    best_map = 0.0
    itr = 0
    model.train()
    if args.loss == 'gss':
        loss_fcn = GSS_loss(args.alpha).gss_loss
    elif args.loss == 'triplet':
        loss_fcn = GSS_loss(args.alpha).tri_loss

    while itr < args.epochs:
        # training step
        start_time = time.time()
        # forward
        for batch_id, batch_data in enumerate(training_loader):
            if args.gpu_id is not None:
                batch_data = batch_data.cuda()
            hidden_emb = model(
                x=training_dataset.features.cuda(
                ) if args.gpu_id is not None else training_dataset.features,
                adj=training_dataset.adj)

            if itr == 0 and batch_id == 0:
                hidden_emb0 = hidden_emb.cpu().data
                if args.beta_percentile is not None:
                    beta_score = np.percentile(
                        np.dot(hidden_emb0, hidden_emb0.transpose(0, 1)).flatten(), args.beta_percentile)
                elif args.beta is not None:
                    beta_score = args.beta
                else:
                    raise Exception(
                        'At least one of beta and beta_percentile should be set!')

            # !# need to change the loss here to support tri_loss and load and use the graph computed
            loss = loss_fcn(embs=hidden_emb, beta=beta_score,
                            index=batch_data.detach())
            # if args.regularizer_scale:
            #     l2 = 0
            #     for p in model.parameters():
            #         l2 += (p**2).sum()
            #     loss += args.regularizer_scale * l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_time = time.time() - start_time
        itr += 1
        # ============
        # eval step
        # ============
        print(f"iter {itr}")
        # print(hidden_emb[0, :10])
    np.savetxt('graph_embs.txt', hidden_emb.cpu().data)


if __name__ == '__main__':
    main(args)
