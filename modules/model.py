from torch.nn.parameter import Parameter
import numpy as np
# from dgl import DGLGraph
# import dgl.function as fn
# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


# from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 g,
                 n_genes,
                 layers=[64],
                 activation=F.relu):
        super().__init__()
        self.g = g  # the graph
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(n_genes*2, layers[0], activation=activation))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(
                GraphConv(layers[i], layers[i+1], activation=activation))
        # output layer
        self.layers.append(GraphConv(layers[-1], n_genes*2))

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense; 
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # h should be (batch, features=2*n_gene)
        h = torch.cat([x_u, x_s], dim=1)  # features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        x = h

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))  # (batch, 64)
        # x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene:2*n_gene]
        pred = beta * x_u + gamma * x_s
        return pred


# implement mini-batch using node-flow in DGL


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)
        # skip connection
        if self.concat:
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNNodeFlow(nn.Module):
    def __init__(self,
                 n_genes,
                 layers=[64],
                 activation=F.relu,
                 dropout=0,
                 **kargs):
        super(GCNNodeFlow, self).__init__()
        self.n_layers = len(layers)
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(
            n_genes*2, layers[0], activation, concat=False))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(NodeUpdate(
                layers[i], layers[i+1], activation, concat=False))
        # output layer
        self.layers.append(NodeUpdate(layers[-1], n_genes*2))

    def forward(self, nf):
        assert nf.layers._graph.num_layers == len(self.layers) + 1
        x_u = nf.layers[0].data['Ux_sz']
        x_s = nf.layers[0].data['Sx_sz']
        batch, n_gene = x_u.shape
        input_activation = torch.cat([x_u, x_s], dim=1)
        # gnn propagate and output x (batch, feature_size)
        x = self._graph_forward(nf, input_activation)

        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene:2*n_gene]
        x_u_out, x_s_out = nf.layers[-1].data['Ux_sz'], nf.layers[-1].data['Sx_sz']
        pred = beta * x_u_out + gamma * x_s_out
        return pred

    def _graph_forward(self, nf, input_activation):
        nf.layers[0].data['activation'] = input_activation

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node: {'h': node.mailbox['m'].mean(dim=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


class GSS_GNNLayer(nn.Module):
    """GNN base layer"""

    def __init__(self, hidden_units, init_weights=1e-5):
        super(GSS_GNNLayer, self).__init__()
        init_w = (np.random.randn(hidden_units, hidden_units) * init_weights)
        init_w[np.where(np.eye(hidden_units) != 0)] = 1

        self.dense = nn.Linear(hidden_units, hidden_units)
        self.dense.weight = Parameter(torch.Tensor(init_w))
        self.dense.bias = Parameter(torch.zeros_like(
            self.dense.bias, dtype=torch.float32))

        self.dense2 = nn.Linear(hidden_units, hidden_units)
        self.dense2.weight = Parameter(torch.Tensor(init_w))
        self.dense2.bias = Parameter(torch.zeros_like(
            self.dense2.bias, dtype=torch.float32))

    def forward(self, features, adj):
        """
        Tensorflow code
        Ax = tf.sparse_tensor_dense_matmul(adj, x)
        pre_nonlinearity = tf.nn.bias_add(tf.matmul(Ax, W), B)
        output = tf.nn.elu(pre_nonlinearity)

        return pre_nonlinearity, output

        """
        # equal to the inter_feature
        Ax = torch.sparse.mm(adj, features)
        # equal to the inter_part1
        pre_nonlinearity1 = self.dense(Ax)

        # inter_part2
        Ax_x = torch.mul(Ax, features)
        Ax_x = torch.sparse.mm(adj, Ax_x)
        pre_nonlinearity2 = self.dense2(Ax_x)

        pre_nonlinearity = pre_nonlinearity1 + pre_nonlinearity2  # /2
        output = F.elu(pre_nonlinearity)

        return pre_nonlinearity, output


class ResidualGraphConvolutionalNetwork(nn.Module):
    def __init__(self, train_batch_size, val_batch_size, num_layers=2,
                 hidden_units=2048, init_weights=1e-5, layer_decay=0.4):
        super(ResidualGraphConvolutionalNetwork, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.init_weights = init_weights
        self.layer_decay = layer_decay
        self.gcn_layer = GSS_GNNLayer(hidden_units, init_weights)

    def decoder(self, x):
        hidden_emb = F.normalize(x, dim=1)
        adj_preds = torch.mm(hidden_emb, hidden_emb.transpose(0, 1))
        adj_preds = F.relu(adj_preds)

        return adj_preds, hidden_emb

    def forward(self, x, adj):
        residual = None
        for i in range(1, self.num_layers + 1):
            pre_nonlinearity, x = self.gcn_layer(x, adj)
            if residual is not None:
                x = residual + self.layer_decay * x
            residual = pre_nonlinearity
        output, hidden_emb = self.decoder(x)

        return output, hidden_emb


class GSS_loss():
    def __init__(self, alpha):
        self.alpha = alpha

    def gss_loss(self, logits, beta, index=None):
        """If index is None, that means computing loss on all data"""
        if index is not None:
            losses = -0.5 * self.alpha * (logits[index][:, index] - beta) ** 2
            return torch.mean(losses)
        losses = -0.5 * self.alpha * (logits - beta) ** 2
        return torch.mean(losses)
