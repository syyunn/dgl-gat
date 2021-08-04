import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

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
            all_head_A[j][i, predecessors] = edge_atten[edges_id, j, 0].data.cpu().numpy()
    if to_normalize:
        for j in range(num_heads):
            all_head_A[j] = normalize(all_head_A[j], norm='l1').tocsr()
    return all_head_A

