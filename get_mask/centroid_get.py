import pickle
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_num', type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--teacher', type=str, default='nnnn')
parser.add_argument('--encode_length', type=int, default=64)
args = parser.parse_args()

load_path = './files/{}_codes_{}.pkl'.format(args.dataset, args.encode_length)
with open(load_path ,'rb') as f:
    data = pickle.load(f)

vectors = []
for k in data.keys():
    vectors.append(data[k])

clf = KMeans(n_clusters=args.cluster_num, n_init=30, init='k-means++')
print("train kmeans model")
clf.fit(vectors)
print("complete!")

file_write_obj = open("./files/kmeans_{}_{}_{}.txt".format(args.dataset, args.cluster_num, args.encode_length), 'w')
for iid, label in enumerate(clf.labels_):
    file_write_obj.writelines(str(iid) + ',' + str(label))
    file_write_obj.write('\n')
file_write_obj.close()

centers = clf.cluster_centers_
with open("./files/kmeans_center_{}_{}_{}.pkl".format(args.dataset, args.cluster_num, args.encode_length), 'wb') as f:
    pickle.dump(centers, f)
