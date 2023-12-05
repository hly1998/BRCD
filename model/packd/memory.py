import torch
from torch import nn
# from torch.nn import functional as F
import math
from queue import deque
import numpy as np


class SimpleMemory(nn.Module):
    """
    a Simple memory bank store all feature in the dataset.
    """
    def __init__(self,opt):
        super(SimpleMemory, self).__init__()
        self.opt = opt
        stdv = 1. / math.sqrt(opt.feat_dim / 3)
        self.register_buffer('memory_bank', torch.rand(
            self.opt.n_data, self.opt.feat_dim).mul_(2 * stdv).add_(-stdv))
        mem_norm = self.memory_bank.pow(2).sum(1, keepdim=True).pow(0.5)
        self.memory_bank = self.memory_bank.div(mem_norm)
        self.memory_change_num = 0

    def get_weights(self, idx):
        weight_mem = torch.index_select(self.memory_bank, 0, idx.view(-1))
        return weight_mem

    def forward(self, feature, y, idx=None, update=False,):
        '''
        input:
          feature: batchSize X inputSize 
          y: 
        output:
          return similarity of input feature and random selected features. 
        '''
        momentum = self.opt.nce_m
        pos_K = self.opt.pos_k
        batchSize = feature.size(0)
        outputSize = self.opt.n_data
        inputSize = self.opt.feat_dim
        # update memory
        with torch.no_grad():
            weight_mem = torch.index_select(self.memory_bank, 0, y.view(-1))
            # check memory bank diff
            if update:
                if self.memory_change_num // 782 == 0:
                    momentum = 0
                # verbose for CIFAR100
                if self.memory_change_num % 782 == 0 or self.memory_change_num % 100 == 0:
                    memory_change_diff = torch.einsum(
                        'ij,ij->i', (weight_mem, feature)).mean().cpu().numpy()
                    print(f'memory_change_diff: epoch {self.memory_change_num // 782}',
                          f'/{self.memory_change_num %782}, {memory_change_diff}',
                          flush=True)
                          
                # update weight
                self.memory_change_num += 1
                weight_mem.mul_(momentum)
                weight_mem.add_(torch.mul(feature, 1 - momentum))
                mem_norm = weight_mem.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_weight = weight_mem.div(mem_norm)
                self.memory_bank.index_copy_(0, y.view(-1), updated_weight)
                return 

        weight = torch.index_select(
            self.memory_bank, 0, idx.contiguous().view(-1)).detach()
        weight = weight.view(batchSize, -1, inputSize)
        output = torch.bmm(weight, feature.view(batchSize, inputSize, 1))
        return output


class DequeMemory(nn.Module):
    '''
    For ImageNet 
    '''
    def __init__(self,opt):
        super(DequeMemory, self).__init__()
        self.opt = opt
        self.deque = deque()
        self.imgid_deque = deque()
        self.max_deque_length = self.opt.n_data # default as 50000
        self.meta_file = self.opt.root_dir + '/meta/train.txt'
        self.load_metas()
        self.nce_k = opt.nce_k

    def load_metas(self):
        with open(self.meta_file) as f:
            lines = f.readlines()
        self.metas = []
        self.data_dict = {}
        self.num_classes = 1000
        for i, line in enumerate(lines):
            filename, label = line.rstrip().split()
            label = int(label)
            self.metas.append({'filename': filename, 'label': label})

    def norm_feats(self, feats, norm=2):
        assert len(feats.shape) == 2, 'feats after transformation must \
        be a {}-dim Tensor.'.format(self.output_dim)
        normed_feats = feats / feats.norm(norm, dim=1, keepdim=True)
        return normed_feats

    def reorg_cls_labeltable(self):
        self.cls_positive = [[] for i in range(self.num_classes)]
        self.cls_negative = [[] for i in range(self.num_classes)]

        dq_imgids = torch.cat(list(self.imgid_deque))
        for i, img_id in enumerate(dq_imgids):
            label = self.metas[img_id]['label']
            self.cls_positive[label].append(i)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i])
                             for i in range(self.num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i])
                             for i in range(self.num_classes)]  

    def forward(self, feature, image_ids, idx=None, update=False,):
        '''
        input:
          feature: batchSize X inputSize 
          y: 
        output:
          return similarity of input feature and random selected features. 
        '''

        batchSize = feature.size(0)
        outputSize = self.opt.n_data
        inputSize = self.opt.feat_dim

        with torch.no_grad():
            if update:
                reserved_t_feats = feature.detach().clone()
                self.deque.append(reserved_t_feats)
                self.imgid_deque.append(image_ids)
                self.reorg_cls_labeltable()
                return
            else:
                n_feats = torch.cat(list(self.deque))
                # select negative samples
                selected_neg_idx = []
                for item in range(batchSize):
                    image_id = image_ids[item]
                    target = self.metas[image_id]['label']
                    replace = True if self.nce_k > len(
                              self.cls_negative[target]) else False
                    neg_idx = np.random.choice(
                            self.cls_negative[target], self.nce_k, replace=replace)                    
                    selected_neg_idx.append(neg_idx)
                selected_neg_idx = np.hstack(selected_neg_idx)
                tn_feats = n_feats[selected_neg_idx].reshape(batchSize, self.nce_k, inputSize)

            # pop out of date samples
            if len(self.deque) >= self.max_deque_length // batchSize:
                self.deque.popleft()
                self.imgid_deque.popleft()

        weight = tn_feats.view(batchSize, -1, inputSize)
        output = torch.bmm(weight, feature.view(batchSize, inputSize, 1))
        return output
        
        
