import time
import numpy as np

import torch
import torch.nn.functional as F

from data_utils import load_cora_data
from model import GAT

g, features, labels, mask = load_cora_data()
num_nodes = g.num_nodes()
num_edges = g.num_edges()

final_layer = 1

g.edata[f'attn_score_l{final_layer}_h{0}'] = torch.zeros(num_edges, 1)

num_classes = len(set(labels.tolist()))
# create the model, 2 heads, each head has hidden size 8
net = GAT(
    g,
    in_dim=features.size()[1],  # 1433
    hidden_dim=8,  # z-dim
    out_dim=num_classes,  # total 7 class labels exists
    num_heads=1,
)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
epochs = 10
for epoch in range(epochs):
    if epoch >= 3:
        t0 = time.time()
    if epoch == epochs - 1:
        save_attn = True
    else:
        save_attn = False
    logits = net(features, save_attn)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)
        )
    )

print(net)

if __name__ == "__main__":
    pass
