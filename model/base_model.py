import math
import torch
import random
import argparse
import torch.nn as nn
from datetime import timedelta
from timeit import default_timer as timer

from utils.data import LabeledData
from utils.evaluation import compress, calculate_top_map

import logging

class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()
    
    def load_data(self):
        self.data = LabeledData(self.hparams.dataset)
    
    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def run_training_sessions(self):
        logging.basicConfig(filename='./logs/' + self.hparams.data_name + '_' + str(self.hparams.trail) + '.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num)
            val_perfs.append(val_perf)
   
        logging.info('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logging.info('best hparams: ' + self.flag_hparams())
        
        val_perf, test_perf = self.run_test()
        logging.info('Val:  {:8.4f}'.format(val_perf))
        logging.info('Test: {:8.4f}'.format(test_perf))
    
    def run_training_session(self, run_num):
        self.train()
        if self.hparams.num_runs > 1:
            logging.info('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)
        
        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()
        if self.hparams.encode_length == 16:
            self.hparams.epochs = max(80, self.hparams.epochs)

        logging.info('hparams: %s' % self.flag_hparams())
        
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)

        optimizer = self.configure_optimizers()
        train_loader, val_loader, _, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        best_val_perf = float('-inf')
        best_state_dict = None
        bad_epochs = 0

        for epoch in range(1, self.hparams.epochs + 1):
            forward_sum = {}
            num_steps = 0
            for batch_num, batch in enumerate(train_loader):
                optimizer.zero_grad()

                imgi, imgj, idxs, _ = batch
                imgi = imgi.to(device)
                imgj = imgj.to(device)

                forward = self.forward(imgi, imgj, device)

                for key in forward:
                    if key in forward_sum:
                        forward_sum[key] += forward[key]
                    else:
                        forward_sum[key] = forward[key]
                num_steps += 1

                if math.isnan(forward_sum['loss']):
                    logging.info('Stopping epoch because loss is NaN')
                    break

                forward['loss'].backward()
                optimizer.step()

            if math.isnan(forward_sum['loss']):
                logging.info('Stopping training session because loss is NaN')
                break
            
            logging.info('End of epoch {:3d}'.format(epoch))
            logging.info(' '.join([' | {:s} {:8.4f}'.format(
                key, forward_sum[key] / num_steps)
                                    for key in forward_sum]))

            if epoch % self.hparams.validate_frequency == 0:
                print('evaluating...')
                val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
                logging.info(' | val perf {:8.4f}'.format(val_perf))

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logging.info('\t\t*Best model so far*')
                    logging.info("saving the best model...")
                    torch.save(self, './checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '.pt')
                else:
                    bad_epochs += 1
                    logging.info('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break

        return None, best_val_perf
    
    def evaluate(self, database_loader, val_loader, topK, device):
        self.eval()
        with torch.no_grad():
            retrievalB, retrievalL, queryB, queryL = compress(database_loader, val_loader, self.encode_discrete, device)
            result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=topK)
        self.train()
        return result

    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        logging.info('load model:' + './checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '.pt')
        self = torch.load('./checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '.pt') if self.hparams.cuda \
                     else torch.load('./checkpoints/' + self.hparams.data_name + '_' + self.hparams.model_name + '_bit:' + str(self.hparams.encode_length) + '.pt', map_location=torch.device('cpu'))
        self.to(device)
    
    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        _, val_loader, test_loader, database_loader = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=False, get_test=True)
        
        val_perf = self.evaluate(database_loader, val_loader, self.data.topK, device)
        test_perf = self.evaluate(database_loader, test_loader, self.data.topK, device)
        return val_perf, test_perf

    def flag_hparams(self):
        flags = '%s' % (self.hparams.data_name)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'data_name', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('data_name', type=str, default='cifar')
        parser.add_argument('model_name', type=str, default='please_choose_a_model')
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('--trail', default = 1, type=int)
        parser.add_argument('-d', '--dataset', default = 'cifar10', type=str,
                            help='dataset [%(default)s]')
        parser.add_argument("-l","--encode_length", type = int, default=16,
                            help = "Number of bits of the hash code [%(default)d]")
        parser.add_argument("--lr", default = 1e-3, type = float,
                            help='initial learning rate [%(default)g]')
        parser.add_argument("--batch_size", default=64,type=int,
                            help='batch size [%(default)d]')
        parser.add_argument("-e","--epochs", default=60, type=int,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=5,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--validate_frequency', type=int, default=5,
                            help='validate every [%(default)d] epochs')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--seed', type=int, default=8888,
                            help='random seed [%(default)d]')
        parser.add_argument('--device', type=int, default=0, 
                            help='device of the gpu')
        
        
        return parser
