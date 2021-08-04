import torch
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import matplotlib.pyplot as plt

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask

def load_graph(path):
    from dgl.data.utils import load_graphs
    glist, label_dict = load_graphs(path)  # glist will be [g1, g2]
    return glist, label_dict


if __name__ == "__main__":
    # g, features, labels, mask = load_cora_data()
    # nx_G = g.to_networkx().to_directed()
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.show()
    glist, label_dict = load_graph("./GAT_20200908125808.bin")
    g = glist[0]
    edge_data_schemes = list(g.edata.keys())
    for scheme in edge_data_schemes:
        if 'attn_score' in scheme:
            attn_score = g.edata[scheme]
    pass
