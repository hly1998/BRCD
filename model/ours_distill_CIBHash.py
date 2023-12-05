# 2023.3.6 更新，我们的模型，见 ours_distill_base_model
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.ours_distill_base_model import Base_Model
import pickle

class CIBHash(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        with open('./get_mask/files/mask_matrix_{}_{}_{}_{}.pkl'.format(self.hparams.margin, self.hparams.cluster_num, self.hparams.dataset, self.hparams.encode_length) ,'rb') as f:
            mask_matrix = pickle.load(f)
        self.mask_matrix = torch.from_numpy(mask_matrix)
        ind2cluster = []
        with open('./get_mask/files/kmeans_{}_{}_{}.txt'.format(self.hparams.dataset, self.hparams.cluster_num, self.hparams.encode_length) ,'r') as f:
            line = f.readline()               # 调用文件的 readline()方法 
            while line: 
                l = int(line.split(',')[1])
                ind2cluster.append(l)
                line = f.readline()
        with open('./get_mask/files/kmeans_center_{}_{}_{}.pkl'.format(self.hparams.dataset, self.hparams.cluster_num, self.hparams.encode_length) ,'rb') as f:
            centers = pickle.load(f)
        self.centers = torch.from_numpy(centers)
        self.ind2cluster = torch.Tensor(ind2cluster).long()
        self.mask_matrix = self.mask_matrix.to(device)
        self.ind2cluster = self.ind2cluster.to(device)
        self.centers = self.centers.to(device)
        if self.hparams.s_model_name == 'mobilenet_v2':
            self.mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
            print("use mobilenet_v2 as backbone")

        if self.hparams.s_model_name == 'resnet18':
            self.resnet = torchvision.models.resnet18(pretrained=True)
            print("use resnet18 as backbone")
            block_num = 1
        if self.hparams.s_model_name == 'resnet34':
            self.resnet = torchvision.models.resnet34(pretrained=True)
            print("use resnet34 as backbone")
            block_num = 1
        if self.hparams.s_model_name == 'resnet50':
            self.resnet = torchvision.models.resnet50(pretrained=True)
            print("use resnet50 as backbone")
            block_num = 4
        if self.hparams.s_model_name == 'resnet101':
            self.resnet = torchvision.models.resnet101(pretrained=True)
            print("use resnet101 as backbone")
            block_num = 4
        if self.hparams.s_model_name == 'resnet152':
            self.resnet = torchvision.models.resnet152(pretrained=True)
            print("use resnet152 as backbone")
            block_num = 4
        
        if self.hparams.s_model_name == 'efficientnet_b0':
            self.efficient_net = torchvision.models.efficientnet_b0(pretrained=True)
            print("use efficientnet_b0 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b1':
            self.efficient_net = torchvision.models.efficientnet_b1(pretrained=True)
            print("use efficientnet_b1 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b2':
            self.efficient_net = torchvision.models.efficientnet_b2(pretrained=True)
            print("use efficientnet_b2 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b3':
            self.efficient_net = torchvision.models.efficientnet_b3(pretrained=True)
            print("use efficientnet_b3 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b4':
            self.efficient_net = torchvision.models.efficientnet_b4(pretrained=True)
            print("use efficientnet_b4 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b5':
            self.efficient_net = torchvision.models.efficientnet_b5(pretrained=True)
            print("use efficientnet_b5 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b6':
            self.efficient_net = torchvision.models.efficientnet_b6(pretrained=True)
            print("use efficientnet_b6 as backbone")
        if self.hparams.s_model_name == 'efficientnet_b7':
            self.efficient_net = torchvision.models.efficientnet_b7(pretrained=True)
            print("use efficientnet_b7 as backbone")

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.fc = nn.Linear(512 * block_num, self.hparams.encode_length)
        
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            for param in self.efficient_net.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(1000, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, self.hparams.encode_length),
                                   )
        if self.hparams.s_model_name in ('mobilenet_v2',):
            for param in self.mobilenet_v2.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(1000, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, self.hparams.encode_length),
                                   )

        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)
        self.criterion_distill = BRCDLoss(self.hparams.batch_size, self.hparams.temperature)
    
    def forward(self, raw_imgi, raw_imgj, idxs, device):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgi = self.resnet(raw_imgi)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgi = self.efficient_net(raw_imgi)
            imgi = self.fc(imgi)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            imgi = self.mobilenet_v2(raw_imgi)
            imgi = self.fc(imgi)
        prob_i = torch.sigmoid(imgi)
        z_i = hash_layer(prob_i - 0.5)

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgj = self.resnet(raw_imgj)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgj = self.efficient_net(raw_imgj)
            imgj = self.fc(imgj)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            imgj = self.mobilenet_v2(raw_imgj)
            imgj = self.fc(imgj)
        prob_j = torch.sigmoid(imgj)
        z_j = hash_layer(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j, device)
        with torch.no_grad():
            t_z_i = self.t_model.encode_discrete(raw_imgi)
            t_z_j = self.t_model.encode_discrete(raw_imgj)
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        fn_matrix = None
        z_i_re = None
        fp_vec = None
        alpha_vec = torch.ones(N) * self.hparams.alpha
        alpha_vec = alpha_vec.to(device)
        if self.hparams.false_neg:
            # f_mask = torch.zeros([N, N])
            cluster_ids = self.ind2cluster[idxs]
            t_matrix = torch.abs(cluster_ids.unsqueeze(-1) - cluster_ids.unsqueeze(0))
            one = torch.ones_like(t_matrix)
            t_matrix = torch.where(t_matrix > 0, one, t_matrix)
            t_matrix = 1 - t_matrix
            t_matrix = t_matrix.repeat(2,2)
            fn_matrix = t_matrix * (-1000)
        if self.hparams.revise_distance:
            cluster_ids = self.ind2cluster[idxs]
            mask_m = self.mask_matrix[cluster_ids]
            mask_m.to(device)
            z_i_re = z_i * mask_m
            t_z_j = t_z_j * mask_m
        if self.hparams.false_pos:
            cluster_ids_for_aug = []
            for iid, c in enumerate(t_z_j):
                aug_c = torch.argmin(torch.sum((c - self.centers)**2, axis=1))
                cluster_ids_for_aug.append(aug_c)
            cluster_ids_for_aug = torch.Tensor(cluster_ids_for_aug).to(device)
            cluster_ids = self.ind2cluster[idxs]
            ii_matrix = torch.abs(cluster_ids.unsqueeze(-1) - cluster_ids.unsqueeze(0))
            ij_matrix = torch.abs(cluster_ids.unsqueeze(-1) - cluster_ids_for_aug.unsqueeze(0))
            # jj_matrix = torch.abs(cluster_ids_for_aug.unsqueeze(-1) - cluster_ids_for_aug.unsqueeze(0))
            up_m = torch.cat((ii_matrix, ij_matrix), dim=1)
            down_m = torch.cat((ii_matrix, ij_matrix), dim=1)
            t_matrix =  torch.cat((up_m, down_m), dim=0)
            one = torch.ones_like(t_matrix)
            t_matrix = torch.where(t_matrix > 0, one, t_matrix)
            t_matrix = 1 - t_matrix
            fn_matrix = t_matrix * (-1000)
            # 获取alpha_vec
            d_ij = torch.diag(ij_matrix)
            for idx, c in enumerate(d_ij):
                if c != 0:
                    alpha_vec[idx] = 1
                    alpha_vec[idx + N//2] = 1
        distll_loss = self.criterion_distill(z_i, t_z_i, t_z_j, alpha_vec, device, z_i_reg=z_i_re, fn_matrix=fn_matrix, fp_vec=fp_vec)
        loss = contra_loss + self.hparams.weight * kl_loss + distll_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}
    
    def encode_discrete(self, x):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            x = self.resnet(x)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            x = self.efficient_net(x)
            x = self.fc(x)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            x = self.mobilenet_v2(x)
            x = self.fc(x)
        prob = torch.sigmoid(x)
        z = hash_layer(prob - 0.5)

        return z

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis = 1))
        return kl

    def configure_optimizers(self):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            return torch.optim.Adam([{'params': self.resnet.fc.parameters()}], lr = self.hparams.lr)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            return torch.optim.Adam([{'params': self.fc.parameters()}], lr = self.hparams.lr)
        if self.hparams.s_model_name in ('mobilenet_v2',):
            return torch.optim.Adam([{'params': self.fc.parameters()}], lr = self.hparams.lr)

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'temperature': [0.2, 0.3, 0.4],
            'weight': [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument('-w',"--weight", default = 0.001, type=float,
                            help='weight of I(x,z) [%(default)f]')
        parser.add_argument("--l2_weight", default = 1, type=float, help='l2部分权重')
        parser.add_argument("--l1_weight", default = 1, type=float, help='l1部分权重')
        parser.add_argument("--kl_distill_weight", default = 1, type=float, help='kl部分权重')
        return parser


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def hash_layer(input):
    return hash.apply(input)


class BRCDLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(BRCDLoss, self).__init__()
        self.temperature = temperature
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.similarityA = nn.CosineSimilarity(dim = 1)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, t_z_i, t_z_j, alpha_vec, device, z_i_reg=None, fn_matrix=None, fp_vec=None):
        """
        z_i: anchor image from student model
        t_z_i: anchor image from teacher model
        t_z_j: augment image from teacher model, it can be mask or not
        z_i_reg: anchor image with mask
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        
        if z_i_reg is not None:
            z = torch.cat((z_i_reg, t_z_j), dim=0)
        else:
            z = torch.cat((z_i, t_z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_anchor = self.similarityA(z_i, t_z_i) / self.temperature
        sim_anchor = torch.cat((sim_anchor, sim_anchor), dim=0)

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)
        positive_samples = alpha_vec * sim_anchor + (1-alpha_vec) * positive_samples
        positive_samples = positive_samples.view(N, 1)
        negative_samples = sim[mask].view(N, -1)
        if fn_matrix is not None:
            false_negative_matrix = fn_matrix[mask].view(N, -1)
            negative_samples = negative_samples + false_negative_matrix
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class NtXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        self.temperature = temperature
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
