from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import SinkhornDistance
from .memory import SimpleMemory, DequeMemory

# from helper.util import AverageMeter, accuracy
eps = 1e-7


class PACKDConLoss(nn.Module):
    def __init__(self, opt, temperature=0.07, ss_T=0.5):
        super(PACKDConLoss, self).__init__()
        self.use_embed = True
        if opt.feat_dim >= opt.t_dim:
            opt.feat_dim = opt.t_dim
            self.use_embed = False
        else:
            self.use_embed = True

        self.temperature = temperature
        print(
            f't_dim: {opt.t_dim} s_dim: {opt.s_dim} feat_dim: {opt.feat_dim} temperature: {self.temperature}')
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim,
                             use_linear=self.use_embed)

        if opt.dataset in ['imagenet']:
            self.memory_t = DequeMemory(opt)
        else:
            self.memory_t = SimpleMemory(opt)
            
        self.iter_num = 0
        self.n_data = opt.n_data
        self.neg_k = opt.nce_k
        self.mixup_num = max(1, opt.mixup_num + 1)
        self.distance_metric = SinkhornDistance(opt.ops_eps, 100, )
        self.ss_T = ss_T
        self.opt = opt

    def forward(self, feat_s, feat_t, labels=None, mask=None, contrast_idx=None,
                mixup_indexes=None, require_feat=False):

        batch_size = feat_s.shape[0] // self.mixup_num
        labels, idx = labels
        embed_s, _ = self.embed_s(feat_s)
        embed_t, _ = self.embed_t(feat_t)

        nor_index = (torch.arange(self.mixup_num*batch_size) %
                     self.mixup_num == 0).cuda()
        aug_index = (torch.arange(self.mixup_num*batch_size) %
                     self.mixup_num != 0).cuda()
        embed_s_nor = embed_s[nor_index]
        embed_s_aug = embed_s[aug_index]
        embed_t_nor = embed_t[nor_index]
        embed_t_aug = embed_t[aug_index]

        if isinstance(contrast_idx, list):
            contrast_idx, pos_idx = contrast_idx
        else:
            pos_idx = None
        # update mempory
        self.memory_t(
            embed_t_nor, idx, contrast_idx, update=True)

        # calculate neg out
        idx = idx.unsqueeze(1).expand(batch_size,
                                        self.mixup_num,
                                        ).reshape(batch_size * self.mixup_num)
        if contrast_idx is not None:
            contrast_idx = contrast_idx.unsqueeze(1).expand(batch_size,
                                                            self.mixup_num,
                                                            -1).reshape(batch_size * self.mixup_num, -1)

        neg_out_s_t = self.memory_t(embed_s, idx, contrast_idx)
        # calculate packd loss
        criterion_packd = ContrastNCELoss(self.n_data, self.mixup_num)
        ident_x = torch.einsum('ij,ij->i', (embed_s, embed_t)).unsqueeze(1)
        # neg_x = neg_out_s_t.narrow(1, 1, self.neg_k).squeeze(2)
        # neg_x = neg_x.reshape(batch_size, -1)
        neg_x = neg_out_s_t.reshape(batch_size, -1)
        embed_s = embed_s.view(batch_size, self.mixup_num, -1)
        embed_t = embed_t.view(batch_size, self.mixup_num, -1)
        cost, pi, C = self.distance_metric(embed_s, embed_t, thresh=self.opt.ops_err_thres)
        pi = pi.detach()
        pos_x = torch.bmm(embed_s, embed_t.transpose(1, 2))
        pos_x = pos_x * pi
        pos_x = pos_x.sum(dim=(-2, -1)).view(batch_size, -1)
        loss_packd = criterion_packd(pos_x,
                                     neg_x,
                                    )
        loss = loss_packd
        # if self.iter_num % 100 == 0 or self.iter_num % 782 == 0:
        #     print(
        #         f'epoch {self.iter_num//782} / {self.iter_num}',
        #         f' loss_packd {loss_packd.item():.4f}',
        #         flush=True)
        self.iter_num += 1
        if require_feat:
            return loss, embed_s.view(batch_size*self.mixup_num,-1)
        else:
            return loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128, use_linear=True, num_classes=100):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.fc = nn.Linear(dim_out, num_classes)
        self.l2norm = Normalize(2)
        self.use_linear = use_linear

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if self.use_linear:
            x = self.linear(x)
        x = self.l2norm(x)
        if self.use_linear:
            logits = self.fc(x * 64)
        else:
            logits = None
        return x, logits

class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ContrastNCELoss(nn.Module):

    def __init__(self, n_data, mixup_num):
        super(ContrastNCELoss, self).__init__()
        self.n_data = n_data
        self.mixup_num = mixup_num

    def forward(self, pos_x, neg_x, pi=None):
        pos_x = torch.exp(torch.div(pos_x, 0.07))
        neg_x = torch.exp(torch.div(neg_x, 0.07))
        bsz = neg_x.shape[0]
        Ng = neg_x.sum(dim=-1, keepdim=True)
        logits = torch.div(pos_x, pos_x.add(Ng)).log_()
        if pi is not None:
            logits = logits * pi
            loss = - logits.sum() / (bsz // self.mixup_num)
        else:
            logits = logits.mean(dim=-1)
            loss = - logits.sum(0) / bsz
        return loss

