import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--margin', type=float, default=0.1)
parser.add_argument('--cluster_num', type=int, default=7)
parser.add_argument('--dataset', type=str)
parser.add_argument('--teacher', type=str, default='nnnn')
parser.add_argument('--encode_length', type=int, default=64)
args = parser.parse_args()

load_path = './files/{}_codes_{}.pkl'.format(args.dataset, args.encode_length)
with open(load_path ,'rb') as f:
    code_data = pickle.load(f)

label_map = {}

with open("./files/kmeans_{}_{}_{}.txt".format(args.dataset, args.cluster_num, args.encode_length) ,'r') as f:
    line = f.readline()               # 调用文件的 readline()方法 
    while line:
        ind = int(line.split(',')[0])
        l = int(line.split(',')[1])
        label_map[ind] = l
        line = f.readline()

code_split = {}
for i in range(args.cluster_num):
    code_split[i] = []
for k in code_data.keys():
    code_split[label_map[k]].append(code_data[k])

matrix = []

for i in range(args.cluster_num):
    mean_bits = np.mean(np.array(code_split[i]), axis=0)
    matrix.append(mean_bits)

matrix = np.array(matrix)
matrix = np.abs(matrix)
matrix = np.where(matrix > args.margin, matrix, 1)
matrix = np.where(matrix < args.margin, matrix, 0)
mask_matrix = matrix
with open('./files/mask_matrix_{}_{}_{}_{}.pkl'.format(args.margin, args.cluster_num, args.dataset, args.encode_length), 'wb') as f:
    pickle.dump(mask_matrix, f)
