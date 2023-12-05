import argparse
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import pickle
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
    if hparams.train:
        torch.cuda.set_device(hparams.device)
        model = CIBHash(hparams)
        model.run_training_sessions()
    else:
        print("**Test**")
        # # 2023.7.13 输出训练数据的hash code用于构建聚类，以及增强数据的code，用于做分析
        # device = torch.device('cuda')
        # # model = torch.load('./checkpoints/cifar_vit_b_16_bit:32.pt', )
        # model = torch.load('./checkpoints/imagenet_vit_b_16_bit:32.pt', )
        # model.to(device)
        # model.eval()
        # print("load ok!")
        # random.seed(6666)
        # torch.manual_seed(6666)
        # device = torch.device('cuda')
        # train_loader, val_loader, test_loader, database_loader = model.data.get_loaders(
        #     32, 2,
        #     shuffle_train=False, get_test=True)
        # codes_dict_i = {}
        # codes_dict_j = {}
        # label_dict = {}
        # cur_id = 0
        # with torch.no_grad():
        #     for batch_step, (imgi, imgj, idx, target) in enumerate(train_loader):
        #         var_data_i = Variable(imgi.to(device))
        #         var_data_j = Variable(imgj.to(device))
        #         i_codes = model.encode_discrete(var_data_i)
        #         j_codes = model.encode_discrete(var_data_j)
        #         for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #             codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #             codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #             cur_id = cur_id + 1
        # # with open("./analysis/distribution/files/cifar10_codes_32.pkl", 'wb') as f:
        # #     pickle.dump(codes_dict_i, f)
        # # with open("./analysis/distribution/files/cifar10_codes_aug_32.pkl", 'wb') as f:
        # #     pickle.dump(codes_dict_j, f)
        # with open("./analysis/distribution/files/imagenet_codes_32.pkl", 'wb') as f:
        #     pickle.dump(codes_dict_i, f)
        # with open("./analysis/distribution/files/imagenet_codes_aug_32.pkl", 'wb') as f:
        #     pickle.dump(codes_dict_j, f)


        # 获取数据库数据的哈希码-数据增强的数据集，注意设置不同seed，以让增强数据未出现过 [更新的版本，只测cifar，但是不同backbone]
        # 2023.9.20 
        # device = torch.device('cuda')
        # t_model = torch.load('./checkpoints/cifar_vit_b_16_bit:64.pt')
        # t_model.to(device)
        # t_model.eval()
        # print("load ok!")
        # random.seed(6666)
        # torch.manual_seed(6666)
        # 跑多轮，等于生成不同的随机数
        # for rand in range(1):
        #     device = torch.device('cuda')
        #     train_loader, val_loader, test_loader, database_loader = t_model.data.get_loaders(
        #         32, 2,
        #         shuffle_train=False, get_test=True)
        #     codes_dict_i = {}
        #     codes_dict_j = {}
        #     label_dict = {}
        #     cur_id = 0
        #     with torch.no_grad():
        #         for imgi, imgj, idxs, target in train_loader:
        #             var_data_i = Variable(imgi.to(device))
        #             var_data_j = Variable(imgj.to(device))
        #             i_codes = t_model.encode_discrete(var_data_i)
        #             j_codes = t_model.encode_discrete(var_data_j)
        #             for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #                 codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #                 codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #                 cur_id = cur_id + 1
        #     with open("./analysis/distribution/files/cifar_vit_b_16_codes_raw_{}.pkl".format(rand), 'wb') as f:
        #         pickle.dump(codes_dict_i, f)
        #     with open("./analysis/distribution/files/cifar_vit_b_16_codes_aug_{}.pkl".format(rand), 'wb') as f:
        #         pickle.dump(codes_dict_j, f)
        #     print("complete[{}/10]".format(rand))

        # 获取数据库数据的哈希码-数据增强的数据集，注意设置不同seed，以让增强数据未出现过。
        # 2023.4.6 cifar数据集的在ours_base_model中已经做了，此处做nuswide数据的
        # if hparams.dataset == 'nuswide':
        #     device = torch.device('cuda')
        #     t_model = torch.load('./checkpoints/nuswide_efficientnet_b3.pt', )
        #     t_model.to(device)
        #     t_model.eval()
        #     print("load ok!")
        #     random.seed(6666)
        #     torch.manual_seed(6666)
        #     device = torch.device('cuda')
        #     train_loader, val_loader, test_loader, database_loader = t_model.data.get_loaders(
        #         32, 2,
        #         shuffle_train=False, get_test=True)
        #     codes_dict_i = {}
        #     codes_dict_j = {}
        #     label_dict = {}
        #     cur_id = 0
        #     with torch.no_grad():
        #         for batch_step, (imgi, imgj, target) in enumerate(train_loader):
        #             var_data_i = Variable(imgi.to(device))
        #             var_data_j = Variable(imgj.to(device))
        #             i_codes = t_model.encode_discrete(var_data_i)
        #             j_codes = t_model.encode_discrete(var_data_j)
        #             for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #                 codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #                 codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #                 cur_id = cur_id + 1
        #     with open("./analysis/distribution/files/nuswide_codes_raw.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_i, f)
        #     with open("./analysis/distribution/files/nuswide_codes_aug.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_j, f)
        # elif hparams.dataset == 'coco':
        #     device = torch.device('cuda')
        #     t_model = torch.load('./checkpoints/coco_efficientnet_b3.pt')
        #     t_model.to(device)
        #     t_model.eval()
        #     print("coco teacher load ok!")
        #     random.seed(6666)
        #     torch.manual_seed(6666)
        #     device = torch.device('cuda')
        #     train_loader, val_loader, test_loader, database_loader = t_model.data.get_loaders(
        #         32, 2,
        #         shuffle_train=False, get_test=True)
        #     codes_dict_i = {}
        #     codes_dict_j = {}
        #     label_dict = {}
        #     cur_id = 0
        #     with torch.no_grad():
        #         for batch_step, (imgi, imgj, target) in enumerate(train_loader):
        #             var_data_i = Variable(imgi.to(device))
        #             var_data_j = Variable(imgj.to(device))
        #             i_codes = t_model.encode_discrete(var_data_i)
        #             j_codes = t_model.encode_discrete(var_data_j)
        #             for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #                 codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #                 codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #                 cur_id = cur_id + 1
        #     with open("./analysis/distribution/files/coco_codes_raw.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_i, f)
        #     with open("./analysis/distribution/files/coco_codes_aug.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_j, f)
        # elif hparams.dataset == 'three':
        #     # 2023.4.14 使用vgg16为teacher model时，这里同时把三个数据集的都给跑了
        #     # cifar
        #     device = torch.device('cuda')
        #     t_model = torch.load('./checkpoints/cifar_vgg16.pt')
        #     t_model.to(device)
        #     t_model.eval()
        #     print("load cifar_vgg16 ok!")
        #     random.seed(6666)
        #     torch.manual_seed(6666)
        #     device = torch.device('cuda')
        #     train_loader, val_loader, test_loader, database_loader = t_model.data.get_loaders(
        #         32, 2,
        #         shuffle_train=False, get_test=True)
        #     codes_dict_i = {}
        #     codes_dict_j = {}
        #     label_dict = {}
        #     cur_id = 0
        #     with torch.no_grad():
        #         for batch_step, (imgi, imgj, target) in enumerate(train_loader):
        #             var_data_i = Variable(imgi.to(device))
        #             var_data_j = Variable(imgj.to(device))
        #             i_codes = t_model.encode_discrete(var_data_i)
        #             j_codes = t_model.encode_discrete(var_data_j)
        #             for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #                 codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #                 codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #                 cur_id = cur_id + 1
        #     with open("./analysis/distribution/files/cifar_vgg16_codes_raw.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_i, f)
        #     with open("./analysis/distribution/files/cifar_vgg16_codes_aug.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_j, f)
        #     print("cifar_vgg16 code output ok!")

        #     # nuswide
        #     t_model = torch.load('./checkpoints/nuswide_vgg16.pt')
        #     t_model.to(device)
        #     t_model.eval()
        #     print("load nuswide_vgg16 ok!")
        #     random.seed(6666)
        #     torch.manual_seed(6666)
        #     device = torch.device('cuda')
        #     train_loader, val_loader, test_loader, database_loader = t_model.data.get_loaders(
        #         32, 2,
        #         shuffle_train=False, get_test=True)
        #     codes_dict_i = {}
        #     codes_dict_j = {}
        #     label_dict = {}
        #     cur_id = 0
        #     with torch.no_grad():
        #         for batch_step, (imgi, imgj, target) in enumerate(train_loader):
        #             var_data_i = Variable(imgi.to(device))
        #             var_data_j = Variable(imgj.to(device))
        #             i_codes = t_model.encode_discrete(var_data_i)
        #             j_codes = t_model.encode_discrete(var_data_j)
        #             for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #                 codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #                 codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #                 cur_id = cur_id + 1
        #     with open("./analysis/distribution/files/nuswide_vgg16_codes_raw.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_i, f)
        #     with open("./analysis/distribution/files/nuswide_vgg16_codes_aug.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_j, f)
        #     print("nuswide_vgg16 code output ok!")

        #     # coco
        #     t_model = torch.load('./checkpoints/coco_vgg16.pt')
        #     t_model.to(device)
        #     t_model.eval()
        #     print("load coco_vgg16 ok!")
        #     random.seed(6666)
        #     torch.manual_seed(6666)
        #     device = torch.device('cuda')
        #     train_loader, val_loader, test_loader, database_loader = t_model.data.get_loaders(
        #         32, 2,
        #         shuffle_train=False, get_test=True)
        #     codes_dict_i = {}
        #     codes_dict_j = {}
        #     label_dict = {}
        #     cur_id = 0
        #     with torch.no_grad():
        #         for batch_step, (imgi, imgj, target) in enumerate(train_loader):
        #             var_data_i = Variable(imgi.to(device))
        #             var_data_j = Variable(imgj.to(device))
        #             i_codes = t_model.encode_discrete(var_data_i)
        #             j_codes = t_model.encode_discrete(var_data_j)
        #             for i_code, j_code, tar in zip(i_codes, j_codes, target):
        #                 codes_dict_i[cur_id] = i_code.data.cpu().numpy().tolist()
        #                 codes_dict_j[cur_id] = j_code.data.cpu().numpy().tolist()
        #                 cur_id = cur_id + 1
        #     with open("./analysis/distribution/files/coco_vgg16_codes_raw.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_i, f)
        #     with open("./analysis/distribution/files/coco_vgg16_codes_aug.pkl", 'wb') as f:
        #         pickle.dump(codes_dict_j, f)
        #     print("coco_vgg16 code output ok!")

