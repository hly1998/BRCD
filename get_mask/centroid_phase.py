import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

if args.dataset == 'cifar10':
    range_list = range(2,15)
    load_path = './files/cifar10_codes_64.pkl'
elif args.dataset == 'coco':
    range_list = range(2,50)
    load_path = './files/coco_codes_64.pkl'
elif args.dataset == 'imagenet':
    load_path = './files/imagenet_codes_64.pkl'
    range_list = range(10,60)

with open(load_path ,'rb') as f:
    data = pickle.load(f)

vectors = []
for k in data.keys():
    vectors.append(data[k])

inertia = []
for cluster_num in range_list:
    clf = KMeans(n_clusters=cluster_num, n_init=30, init='k-means++')
    clf.fit(vectors)
    inertia.append(clf.inertia_)
plt.plot(range_list, inertia, 'o-')
plt.savefig('./images/kmeans_{}_elbow.png'.format(args.dataset))

silhouette_scores = []
for cluster_num in range_list:
    clf = KMeans(n_clusters=cluster_num, n_init=30, init='k-means++')
    clf.fit(vectors)
    silhouette_scores.append(metrics.silhouette_score(vectors, clf.labels_ , metric='euclidean')) 
plt.plot(range_list, silhouette_scores, 'o-')
plt.savefig('./images/kmeans_{}_silhouette.png'.format(args.dataset))