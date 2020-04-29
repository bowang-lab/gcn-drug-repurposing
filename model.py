import torch
import torch.nn as nn
import torch.nn.functional as F


from dgl.nn.pytorch import GraphConv
class GCN(nn.Module):
    def __init__(self,
                 g,
                 n_genes,
                 layers = [64],
                 activation =F.relu):
        super().__init__()
        self.g = g # the graph  
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(n_genes*2, layers[0], activation=activation))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(GraphConv(layers[i], layers[i+1], activation=activation))
        # output layer
        self.layers.append(GraphConv(layers[-1], n_genes*2))

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense; 
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # h should be (batch, features=2*n_gene)
        h = torch.cat([x_u, x_s], dim=1) # features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        x = h

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))  # (batch, 64)
        # x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene] # (batch, genes)
        gamma = x[:, n_gene:2*n_gene]
        pred = beta * x_u + gamma * x_s
        return pred


# implement mini-batch using node-flow in DGL
import dgl
import dgl.function as fn
from dgl import DGLGraph
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
        self.layers.append(NodeUpdate(n_genes*2, layers[0], activation, concat=False))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(NodeUpdate(layers[i], layers[i+1], activation, concat=False))
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

        beta = x[:, 0:n_gene] # (batch, genes)
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
                             lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h