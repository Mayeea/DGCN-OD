import torch.nn as nn
import torch.nn.functional as F

import torch
from mp_deterministic import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag
from torch.nn import Module, ModuleList, Linear, LayerNorm

class GatingGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, hidden_channel, chunk_size,
                 add_self_loops=True, tm=True, simple_gating=False,
                 diff_or=True):
        super(GatingGNNConv, self).__init__('mean')
        self.tm_net = tm_net
        self.tm_norm = tm_norm
        self.tm = tm
        self.diff_or = diff_or
        self.simple_gating = simple_gating
        self.hidden_channel = hidden_channel
        self.chunk_size = chunk_size

    def forward(self, x, edge_index, last_tm_signal):
        print(f"Input x dimensions: {x.size()}")
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if add_self_loops==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if add_self_loops==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        print(f"Aggregated message m dimensions: {m.size()}")
        if self.tm==True:
            if self.simple_gating==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
                print("Applying simple gating.")
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                print("Applying complex gating with cumulative sum.")
                if self.diff_or==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw

            print(f"tm_signal_raw dimensions: {tm_signal_raw.size()}")
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.hidden_channel/self.chunk_size), dim=1)
            print(f"Repeated tm_signal dimensions: {tm_signal.size()}")
            out = x*tm_signal + m*(1-tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)
        print(f"Output out dimensions: {out.size()}")

        return out, tm_signal_raw

class GatingGNN(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers_input=1,
                 global_gating=True, num_layers=2, dropout_rate=0.4, dropout_rate2=0.4):
        super().__init__()
        self.linear_trans_in = ModuleList()
        self.linear_trans_out = Linear(hidden_channel, out_channel)
        self.norm_input = ModuleList()
        self.convs = ModuleList()
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate = dropout_rate
        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()
        self.chunk_size = 64
        self.linear_trans_in.append(Linear(in_channel, hidden_channel))

        self.norm_input.append(LayerNorm(hidden_channel))

        for i in range(num_layers_input - 1):
            self.linear_trans_in.append(Linear(hidden_channel, hidden_channel))
            self.norm_input.append(LayerNorm(hidden_channel))

        if global_gating == True:
            tm_net = Linear(2 * hidden_channel, self.chunk_size)

        for i in range(num_layers):
            self.tm_norm.append(LayerNorm(hidden_channel))

            if global_gating == False:
                self.tm_net.append(Linear(2 *hidden_channel, self.chunk_size))
            else:
                self.tm_net.append(tm_net)
            self.convs.append(GatingGNNConv(tm_net=self.tm_net[i], tm_norm=self.tm_norm[i], hidden_channel=hidden_channel, chunk_size=self.chunk_size))

        self.params_conv = list(set(list(self.convs.parameters()) + list(self.tm_net.parameters())))
        self.params_others = list(self.linear_trans_in.parameters()) + list(self.linear_trans_out.parameters())

    def forward(self, x, edge_index):
        check_signal = []
        print(f"Initial input x dimensions: {x.size()}")

        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(self.linear_trans_in[i](x))
            x = self.norm_input[i](x)
            print(f"After linear transformation {i}, x dimensions: {x.size()}")

        tm_signal = x.new_zeros(self.chunk_size)

        for j in range(len(self.convs)):
            if self.dropout_rate2 != 'None':
                x = F.dropout(x, p=self.dropout_rate2, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout_rate2, training=self.training)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)
            print(f"After convolution {j}, x dimensions: {x.size()}, tm_signal dimensions: {tm_signal.size()}")
            check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear_trans_out(x)
        print(f"Final output x dimensions: {x.size()}")
        encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        return encode_values['x']


class ComGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(ComGCN, self).__init__()
        self.GatingGNN = GatingGNN(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        if fadj.is_sparse:
            newfadj = fadj.to_dense()
            edge_index = newfadj.nonzero(as_tuple=True)
            edge_index = torch.stack(edge_index, dim=0)
        if sadj.is_sparse:
            newsadj = sadj.to_dense()
            sedge_index = newsadj.nonzero(as_tuple=True)
            sedge_index = torch.stack(sedge_index, dim=0)
            com1 = self.GONN(x, sedge_index)
            com2 = self.GONN(x, edge_index)

        Xcom = (com1 + com2) / 2
        emb = Xcom
        output = self.MLP(emb)
        return output, com1, com2, emb