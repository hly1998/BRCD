import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.distill_base_model import Base_Model
from model.crd.criterion import CRDLoss
from model.crcd.criterion import CRCDLoss
from model.packd.packd import PACKDConLoss


class CIBHash(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.kl_criterion = DistillKL()
 
    def define_parameters(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        if self.hparams.s_model_name == 'efficientnet_b0':
            self.efficient_net = torchvision.models.efficientnet_b0(pretrained=True)
            print("use efficientnet_b0 as backbone")
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            for param in self.efficient_net.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(1000, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, self.hparams.encode_length),
                                   )
        if self.hparams.rkd:
            self.kd_criterion = RKDLoss()
        elif self.hparams.pkt:
            self.kd_criterion = PKT()
        elif self.hparams.sp:
            self.kd_criterion = Similarity()
        elif self.hparams.crd:
            print("using crd distillation method....")
            self.opt = type('', (), {})()
            self.opt.embed_type = 'linear'
            self.opt.s_dim = self.hparams.encode_length
            self.opt.t_dim = self.hparams.encode_length
            self.opt.feat_dim = 128
            self.opt.nce_k = 500
            self.opt.nce_t = 0.05
            self.opt.nce_m = 0.5
            self.opt.n_data = 5000
            self.kd_criterion = CRDLoss(self.opt).to(device)
        elif self.hparams.sskd:
            self.kd_criterion = DistillSSKD()
        elif self.hparams.crcd:
            self.opt = type('', (), {})()
            self.opt.embed_type = 'linear'
            self.opt.s_dim = self.hparams.encode_length
            self.opt.t_dim = self.hparams.encode_length
            self.opt.feat_dim = 128
            self.opt.nce_k = 500
            self.opt.nce_t = 0.05
            self.opt.nce_m = 0.5
            self.opt.n_data = 5000
            self.criterion_kd = CRCDLoss(self.opt).to(device)
        elif self.hparams.packd:
            self.opt = type('', (), {})()
            self.opt.s_dim = self.hparams.encode_length
            self.opt.t_dim = self.hparams.encode_length
            self.opt.feat_dim = 64
            self.opt.nce_k = 500
            self.opt.pos_k = -1
            self.opt.nce_m = 0.5
            self.opt.mixup_num = 0
            self.opt.dataset = self.hparams.dataset
            self.opt.n_data = 5000
            self.opt.ops_eps = 0.1
            self.opt.ops_err_thres = 0.1
            self.kd_criterion = PACKDConLoss(self.opt)
        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)
    
    def forward(self, raw_imgi, raw_imgj, device, index=None, contrast_idx=None, distill_img=None, img=None, labels=None, mixup_indexes=None):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgi = self.resnet(raw_imgi)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgi = self.efficient_net(raw_imgi)
            imgi = self.fc(imgi)
        prob_i = torch.sigmoid(imgi)
        z_i = hash_layer(prob_i - 0.5)

        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            imgj = self.resnet(raw_imgj)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgj = self.efficient_net(raw_imgj)
            imgj = self.fc(imgj)
        prob_j = torch.sigmoid(imgj)
        z_j = hash_layer(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j, device)
        with torch.no_grad():
            t_z_i = self.t_model.encode_discrete(raw_imgi)
        if self.hparams.rkd:
            distll_loss = self.hparams.l1_weight * self.kd_criterion(z_i, t_z_i)
        elif self.hparams.pkt:
            distll_loss = self.hparams.l1_weight * self.kd_criterion(z_i, t_z_i)
        elif self.hparams.sp:
            distll_loss = self.hparams.l1_weight * sum(self.kd_criterion(z_i, t_z_i))[0]
        elif self.hparams.crd:
            distll_loss = self.kd_criterion(z_i, t_z_i, index, contrast_idx)[0]
        elif self.hparams.sskd:
            c,h,w = distill_img.size()[-3:]
            input_x = distill_img.view(-1,c,h,w).cuda()
            batch = int(input_x.size(0) / 4)
            x = self.efficient_net(input_x)
            x = self.fc(x)
            x = torch.sigmoid(x)
            x = hash_layer(x - 0.5)
            xt = self.t_model.encode_discrete(input_x)
            distll_loss = self.kd_criterion(x, xt, batch)
        elif self.hparams.crcd:
            distll_loss = self.kd_criterion(z_i, t_z_i, index, contrast_idx)
        elif self.hparams.packd:
            mixup_num = 1
            c, h, w = img.size()[-3:]
            input_x = img.view(-1, c, h, w).cuda()
            batch = int(img.size(0) // mixup_num)
            x = self.efficient_net(input_x)
            x = self.fc(x)
            x = torch.sigmoid(x)
            x = hash_layer(x - 0.5)
            xt = self.t_model.encode_discrete(input_x)
            distll_loss = self.kd_criterion(x, xt, labels=labels,
                                       mask=None,
                                       contrast_idx=contrast_idx,
                                       mixup_indexes=mixup_indexes)
        elif self.hparams.kl:
            distll_loss = self.hparams.kl_distill_weight * self.kl_criterion(z_i, t_z_i)
        loss = contra_loss + self.hparams.weight * kl_loss + distll_loss
        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}
    
    def encode_discrete(self, x):
        if self.hparams.s_model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            x = self.resnet(x)
        if self.hparams.s_model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            x = self.efficient_net(x)
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
        parser.add_argument("--l2_weight", default = 1, type=float)
        parser.add_argument("--l1_weight", default = 1, type=float)
        parser.add_argument("--kl_distill_weight", default = 1, type=float)
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

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning 2018
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss

class Similarity(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author
    SP distillation
    """
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network 2015"""
    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class DistillSSKD(nn.Module):
    """
    SSKD loss
    """
    def __init__(self):
        super(DistillSSKD, self).__init__()

    def forward(self, s_feat, t_feat, batch):
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()
        s_nor_feat = s_feat[nor_index]
        s_aug_feat = s_feat[aug_index]
        s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)
        t_nor_feat = t_feat[nor_index]
        t_aug_feat = t_feat[aug_index]
        t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)
        t_simi = t_simi.detach()
        aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        rank = torch.argsort(t_simi, dim=1, descending=True)
        rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        index = torch.argsort(rank)
        tmp = torch.nonzero(rank, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = 3*batch - wrong_num
        wrong_keep = int(wrong_num * 1.0)
        index = index[:correct_num+wrong_keep]
        distill_index_ss = torch.sort(index)[0]
        # ss_T = 0.5
        log_simi = F.log_softmax(s_simi / 0.5, dim=1)
        simi_knowledge = F.softmax(t_simi / 0.5, dim=1)
        distll_loss = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], reduction='batchmean') * 0.5 * 0.5
        return distll_loss