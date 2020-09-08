import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, nth_layer, ith_head, final_layer):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()
        self.nth_layer = nth_layer
        self.ith_head = ith_head
        self.final_layer = final_layer
        self.save_attn = False

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # print(edges.data["e"].shape)
        # edges_tuple = edges.edges()  #  src[i],dst[i],eid[i]
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        # save attention scores @ final
        if self.nth_layer == self.final_layer and self.save_attn:
            print("assigning final layer's attention..")
            dsc_ids = nodes.nodes().tolist()
            for i, dsc_id in enumerate(dsc_ids):
                # print(f"Assigning dsc_id .. for {dsc_id}")
                src_ids = self.g.predecessors(dsc_id)
                # print(alpha[i, :, :].sum())
                self.g.edata[f'attn_score_l{self.nth_layer}_h{self.ith_head}'][self.g.edge_ids(src_ids, dsc_id)] = alpha[i, :, :]  # edges [1, 2, 3] -> 0 # num_head is 1 in case of final layer
        return {"h": h}

    def forward(self, h, save_attn):
        self.save_attn = save_attn
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, nth_layer, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim, nth_layer=nth_layer, ith_head=i, final_layer=1))
        self.merge = merge

    def forward(self, h, save_attn):
        head_outs = [attn_head(h, save_attn) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads, nth_layer=0)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, num_heads=1, nth_layer=1)

    def forward(self, h, save_attn):
        h = self.layer1(h, save_attn)
        h = F.elu(h)
        h = self.layer2(h, save_attn)
        return h


if __name__ == "__main__":
    pass
