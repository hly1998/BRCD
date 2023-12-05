# import argparse
import torch
# import torch.nn as nn
import random
# from torch.autograd import Variable
# import pickle
from model.CIBHash import CIBHash
import psutil

count = psutil.cpu_count()
print(f"the number of logit cpu is{count}")
p = psutil.Process()
p.cpu_affinity(list(random.sample(range(1, count), 6)))
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    argparser = CIBHash.get_model_specific_argparser()
    hparams = argparser.parse_args()
    torch.cuda.set_device(hparams.device)
    model = CIBHash(hparams)
    model.run_training_sessions()
       