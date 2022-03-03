
import torch
import torch.nn as nn
import pdb
import copy
import torch.nn.functional as F



def Hinge_loss(f, psd_label, margin):
    N = f.size(0)
    f = F.normalize(f, p=2, dim=-1)
    psd_label = F.normalize(psd_label, p=2, dim=-1)
    l_f_product = psd_label.matmul(f.t())
    l_f_diag = torch.diag(l_f_product).unsqueeze(0)
    l_f_d = l_f_product - l_f_diag
    L = (torch.sum(F.relu(l_f_d + margin)) - N*margin) / (N**2 - N)

    return L


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  # 相当于一个list

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_a):
        return x + self.dropout(x_a)



class TR_module(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TR_module, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.sublayer = SublayerConnection(d_model, dropout)
        feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.feed_forward = feed_forward

    def forward(self, f, word2vec, tag_inds):
        pad_masks = (tag_inds == 0)
        _, attn_w1 = self.attn(f.transpose(0,1), word2vec.transpose(0,1), word2vec.transpose(0,1))  
        t1, attn_w2 = self.attn(f.transpose(0,1), word2vec.transpose(0,1), word2vec.transpose(0,1), key_padding_mask=pad_masks)
        
        attn_w1, attn_w2 = attn_w1.squeeze(), attn_w2.squeeze()

        t1 = t1.squeeze()
        t2 = self.feed_forward(t1)
        t2 = self.sublayer(t1, t2)

        return t2, attn_w1, attn_w2



















