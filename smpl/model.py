import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(
        self,
        g,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


from scipy.sparse import lil_matrix


def preprocess_attention(edge_atten, g, to_normalize=True):
    """Organize attentions in the form of csr sparse adjacency
    matrices from attention on edges.

    Parameters
    ----------
    edge_atten : numpy.array of shape (# edges, # heads, 1)
        Un-normalized attention on edges.
    g : dgl.DGLGraph.
    to_normalize : bool
        Whether to normalize attention values over incoming
        edges for each node.
    """
    n_nodes = g.number_of_nodes()
    num_heads = edge_atten.shape[1]
    all_head_A = [lil_matrix((n_nodes, n_nodes)) for _ in range(num_heads)]
    for i in range(n_nodes):
        predecessors = list(g.predecessors(i))
        edges_id = g.edge_ids(predecessors, i)
        for j in range(num_heads):
            all_head_A[j][i, predecessors] = (
                edge_atten[edges_id, j, 0].data.cpu().numpy()
            )
    if to_normalize:
        for j in range(num_heads):
            all_head_A[j] = normalize(all_head_A[j], norm="l1").tocsr()
    return all_head_A


# # Take the attention from one layer as an example
# # num_edges x num_heads x 1
# A = self.g.edata["a_drop"]
# # list of length num_heads, each entry is csr of shape (num_nodes, num_nodes)
# A = preprocess_attention(A, self.g)

if __name__ == "__main__":
    pass
