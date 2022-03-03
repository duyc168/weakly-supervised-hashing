
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from Modules.layers import TR_module


class retrieval_model(nn.Module):
    def __init__(self, model):
        super(retrieval_model, self).__init__()
        self.feature = model.feature
        self.classifier = model.classifier
        self.H1 = model.H1
        self.fc3 = model.fc3

    def forward(self, x):
        x = self.feature(x)
        f = x.view(x.size(0), 6*6*256)
        f = self.classifier(f)
        h = self.H1(f)
        fr = self.fc3(h)

        return h, fr



class WSD_model(nn.Module):
    def __init__(self, original_model, d_model, heads, d_ff, bit):
        super(WSD_model, self).__init__()
        self.feature = original_model.features

        cl1 = nn.Linear(256*6*6, 4096)
        cl1.weight = original_model.classifier[1].weight
        cl1.bias = original_model.classifier[1].bias
        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[4].weight
        cl2.bias = original_model.classifier[4].bias
        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2, 
            nn.ReLU(inplace=True),
        )

        self.H1 = nn.Sequential(
            nn.Linear(4096, bit),
            nn.Sigmoid(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(bit, d_model),
            nn.LeakyReLU(inplace=True),
        )

        for m in self.fc3.children():
            if isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_normal_(m.weight)
        
        self.TR_module = TR_module(d_model, heads, d_ff, 0.1)


    def forward(self, x, word2vec, tag_inds):
        x = self.feature(x)
        x = x.view(x.size(0), 6*6*256)
        x = self.classifier(x)
        h = self.H1(x)
        f = self.fc3(h)

        f1 = f.view(f.size(0), 1, f.size(1))
        word2vec = word2vec.view(1, word2vec.size(0), word2vec.size(1))
        word2vec = word2vec.expand(x.size(0), word2vec.size(1), word2vec.size(2))
        psd_label, attn_w1, attn_w2 = self.TR_module(f1, word2vec, tag_inds)

        return h, f, psd_label, attn_w1, attn_w2