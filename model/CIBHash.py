import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Function

from model.base_model import Base_Model

class CIBHash(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        if self.hparams.model_name == 'vit_b_16':
            self.net = torchvision.models.vit_b_16(pretrained=True)
            self.net.heads = nn.Linear(768, self.hparams.encode_length)
        if self.hparams.model_name == 'vit_b_32':
            self.net = torchvision.models.vit_b_32(pretrained=True)
        if self.hparams.model_name == 'vit_h_14':
            self.net = torchvision.models.vit_h_14(weights='IMAGENET1K_V1')
        if self.hparams.model_name == 'vit_l_16':
            self.net = torchvision.models.vit_l_16(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])   
        if self.hparams.model_name == 'efficientnet_b0':
            self.efficient_net = torchvision.models.efficientnet_b0(pretrained=True)
            print("use efficientnet_b0 as backbone")
        print("use {} as backbone".format(self.hparams.model_name))
        if self.hparams.model_name in ('vit_b_16','vit_b_32','vit_l_16','vit_h_14'):
            for name, param in self.net.named_parameters():
                if 'heads' not in name:
                    param.requires_grad = False
        if self.hparams.model_name in ('vgg16',):
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.encoder = nn.Sequential(nn.Linear(4096, 1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, self.hparams.encode_length),
                                        )        
        if self.hparams.model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            for param in self.efficient_net.parameters():
                param.requires_grad = False
            self.fc = nn.Linear(1000, self.hparams.encode_length)

        self.criterion = NtXentLoss(self.hparams.batch_size, self.hparams.temperature)
    
    def forward(self, imgi, imgj, device):
        if self.hparams.model_name in ('vit_b_16','vit_b_32','vit_l_16','vit_h_14'):
            imgi = self.net(imgi)
            imgj = self.net(imgj)

        if self.hparams.model_name in ('vgg16',):
            imgi = self.vgg.features(imgi)
            imgi = imgi.view(imgi.size(0), -1)
            imgi = self.vgg.classifier(imgi)
            imgi = self.encoder(imgi)
            imgj = self.vgg.features(imgj)
            imgj = imgj.view(imgj.size(0), -1)
            imgj = self.vgg.classifier(imgj)
            imgj = self.encoder(imgj)
        if self.hparams.model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            imgi = self.efficient_net(imgi)
            imgi = self.fc(imgi)
            imgj = self.efficient_net(imgj)
            imgj = self.fc(imgj)
        
        prob_i = torch.sigmoid(imgi)
        z_i = hash_layer(prob_i - 0.5)
        prob_j = torch.sigmoid(imgj)
        z_j = hash_layer(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j, device)
        loss = contra_loss + self.hparams.weight * kl_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}
    
    def encode_discrete(self, x):
        if self.hparams.model_name in ('vit_b_16','vit_b_32','vit_h_14','vit_l_16'):
            x = self.net(x)
        if self.hparams.model_name in ('vgg16',):
            x = self.vgg.features(x)
            x = x.view(x.size(0), -1)
            x = self.vgg.classifier(x)
            x = self.encoder(x)
        if self.hparams.model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            x = self.resnet(x)
        if self.hparams.model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
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
        if self.hparams.model_name in ('vit_b_16','vit_b_32','vit_h_14','vit_l_16'):
            return torch.optim.Adam([{'params': self.net.heads.parameters()}], lr = self.hparams.lr)
        if self.hparams.model_name in ('vgg16',):
            return torch.optim.Adam([{'params': self.encoder.parameters()}], lr = self.hparams.lr)
        if self.hparams.model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            return torch.optim.Adam([{'params': self.resnet.fc.parameters()}], lr = self.hparams.lr)
        if self.hparams.model_name in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'):
            return torch.optim.Adam([{'params': self.fc.parameters()}], lr = self.hparams.lr)

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument("-t", "--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument('-w',"--weight", default = 0.001, type=float,
                            help='weight of I(x,z) [%(default)f]')
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