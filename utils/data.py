import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
from utils.gaussian_blur import GaussianBlur
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_datasets()

        # setup dataTransform
        color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
        self.train_transforms = transforms.Compose([transforms.RandomResizedCrop(size = 224,scale=(0.5, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p = 0.7),
                                            transforms.RandomGrayscale(p  = 0.2),
                                            GaussianBlur(3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                            ])
        self.test_transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                         
        ])
        self.test_cifar10_transforms = transforms.Compose([
                                            transforms.Resize((224, 224)),  
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                         
        ])


    
    def load_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        train_dataset = MyTrainDataset(self.X_train, self.Y_train, self.train_transforms, self.dataset)

        if(self.dataset == 'cifar10'):
            val_dataset = MyTestDataset(self.X_val, self.Y_val, self.test_cifar10_transforms, self.dataset)
            test_dataset = MyTestDataset(self.X_test, self.Y_test, self.test_cifar10_transforms, self.dataset)
            database_dataset = MyTestDataset(self.X_database, self.Y_database, self.test_cifar10_transforms, self.dataset)
        else:
            val_dataset = MyTestDataset(self.X_val, self.Y_val, self.test_transforms, self.dataset)
            test_dataset = MyTestDataset(self.X_test, self.Y_test, self.test_transforms, self.dataset)
            database_dataset = MyTestDataset(self.X_database, self.Y_database, self.test_transforms, self.dataset)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                shuffle=shuffle_train,
                                                num_workers=num_workers)

        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers)

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers) if get_test else None

        database_loader = DataLoader(dataset=database_dataset, batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)
        
        return train_loader, val_loader, test_loader, database_loader

class LabeledData(Data):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)
    
    def load_datasets(self):
        if(self.dataset == 'cifar10'):
            self.topK = 1000
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_cifar()
        elif self.dataset == 'nuswide':
            self.root = '[the root of nuswide]'
            self.topK = 5000
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_nuswide()
        elif self.dataset == 'coco':
            self.topK = 5000
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_coco()
        elif self.dataset == 'imagenet':
            self.topK = 1000
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.X_database, self.Y_database = get_imagenet()
        else:
            raise NotImplementedError("Please use the right dataset!")

class MyTrainDataset(Dataset):
    def __init__(self, data,labels, transform, dataset):
        self.data = data
        self.labels = labels
        self.transform  = transform
        self.dataset = dataset
        if self.dataset == 'nuswide':
            self.root = '[the root of nuswide]'
        if self.dataset == 'imagenet':
            self.root = '[the root of imagenet]'
    def __getitem__(self, index):
        # print("dataset:", self.dataset)
        if self.dataset == 'cifar10':
            pilImg = Image.fromarray(self.data[index])
            imgi = self.transform(pilImg)
            imgj = self.transform(pilImg)
            return (imgi, imgj, index, self.labels[index])
        elif self.dataset == 'nuswide':
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            imgi = self.transform(img)
            imgj = self.transform(img)
            return (imgi, imgj, index, self.labels[index])
        elif self.dataset == 'coco':
            img = Image.open(self.data[index]).convert('RGB')
            imgi = self.transform(img)
            imgj = self.transform(img)
            return (imgi, imgj, index, self.labels[index])
        elif self.dataset == 'imagenet':
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            imgi = self.transform(img)
            imgj = self.transform(img)
            return (imgi, imgj, index, self.labels[index])

    def __len__(self):
        return len(self.data)

class MyTestDataset(Dataset):
    def __init__(self,data,labels, transform, dataset):
        self.data = data
        self.labels = labels
        self.transform  = transform
        self.dataset = dataset
        if self.dataset == 'nuswide':
            self.root = './data/nuswide/NUS-WIDE'
        if self.dataset == 'imagenet':
            self.root = '/data/lyhe/BNN/DeepHash-pytorch-master/data/imagenet'
    def __getitem__(self, index):
        if self.dataset == 'cifar10':
            pilImg = Image.fromarray(self.data[index])
            return (self.transform(pilImg), index, self.labels[index])
        elif self.dataset == 'nuswide':
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            return (self.transform(img), index, self.labels[index])
        elif self.dataset == 'coco':
            img = Image.open(self.data[index]).convert('RGB')
            return (self.transform(img), index, self.labels[index])
        elif self.dataset == 'imagenet':
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            return (self.transform(img), index, self.labels[index])
        
    def __len__(self):
        return len(self.data)

def get_cifar():
    # Dataset
    train_dataset = dsets.CIFAR10(root='./data/cifar10/',
                                train=True,
                                download=True)

    test_dataset = dsets.CIFAR10(root='./data/cifar10/',
                                train=False
                                )

    database_dataset = dsets.CIFAR10(root='./data/cifar10/',
                                    train=True
                                    )


    # train with 5000 images
    X = train_dataset.data
    L = np.array(train_dataset.targets)

    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        N = index.shape[0]
        prem = np.random.permutation(N)
        index = index[prem]
        
        data = X[index[0:500]]
        labels = L[index[0: 500]]
        if first:
            Y_train = labels
            X_train = data
        else:
            Y_train = np.concatenate((Y_train, labels))
            X_train = np.concatenate((X_train, data))
        first = False

    Y_train = np.eye(10)[Y_train]

    
    idxs = list(range(len(test_dataset.data)))
    np.random.shuffle(idxs)
    test_data = np.array(test_dataset.data)
    test_tragets = np.array(test_dataset.targets)

    X_val = test_data[idxs[:5000]]
    Y_val = np.eye(10)[test_tragets[idxs[:5000]]]

    X_test = test_data[idxs[5000:]]
    Y_test = np.eye(10)[test_tragets[idxs[5000:]]]


    X_database = database_dataset.data 
    Y_database = np.eye(10)[database_dataset.targets]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database


def get_nuswide(num_train=10500):
    root = './data/nuswide/NUS-WIDE'
    # training dataset
    img_txt_path = os.path.join(root, 'database_img.txt')
    label_txt_path = os.path.join(root, 'database_label_onehot.txt')
    with open(img_txt_path, 'r') as f:
        data = np.array([i.strip() for i in f])
    targets = np.loadtxt(label_txt_path, dtype=np.float32)
    perm_index = np.random.permutation(len(data))[:num_train]
    X_train = data[perm_index]
    Y_train = targets[perm_index]
    # print("X_train", X_train, type(X_train), X_train.shape)
    # print("Y_train", Y_train, type(Y_train), Y_train.shape)
    # exit()
    # test and valid dataset
    img_txt_path = os.path.join(root, 'test_img.txt')
    label_txt_path = os.path.join(root, 'test_label_onehot.txt')
    with open(img_txt_path, 'r') as f:
        data = np.array([i.strip() for i in f])
    targets = np.loadtxt(label_txt_path, dtype=np.float32)
    X_val = data
    Y_val = targets
    # val = test
    l = X_val.shape[0]
    X_test = data
    Y_test = targets
    # print("X_test", X_test, type(X_test), X_test.shape)
    # print("Y_test", Y_test, type(Y_test), Y_test.shape)
    # exit()
    # database dataset
    img_txt_path = os.path.join(root, 'database_img.txt')
    label_txt_path = os.path.join(root, 'database_label_onehot.txt')
    with open(img_txt_path, 'r') as f:
        data = np.array([i.strip() for i in f])
    targets = np.loadtxt(label_txt_path, dtype=np.float32)
    X_database = data
    Y_database = targets
    print("Load NUS-WIDE dataset complete...")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database

def get_coco():
    root = './data/coco/'
    base_folder = 'train.txt'
    data = []
    labels = []
    num_classes = 80
    filename = os.path.join(root, base_folder)
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pos_tmp = lines.split()[0]
            pos_tmp = pos_tmp[37:]
            pos_tmp = os.path.join(root, pos_tmp)
            label_tmp = lines.split()[1:]
            data.append(pos_tmp)
            labels.append(label_tmp)
    data = np.array(data)
    labels = np.float64(labels)
    labels.reshape((-1, num_classes))
    X_train = data
    Y_train = labels
    base_folder = 'test.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pos_tmp = lines.split()[0]
            pos_tmp = pos_tmp[37:]                
            pos_tmp = os.path.join(root, pos_tmp)
            label_tmp = lines.split()[1:]
            data.append(pos_tmp)
            labels.append(label_tmp)
    data = np.array(data)
    labels = np.float64(labels)
    labels.reshape((-1, num_classes))
    X_val = data
    Y_val = labels
    X_test = data
    Y_test = labels
    base_folder = 'database.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pos_tmp = lines.split()[0]
            pos_tmp = pos_tmp[37:]
            pos_tmp = os.path.join(root, pos_tmp)
            label_tmp = lines.split()[1:]
            data.append(pos_tmp)
            labels.append(label_tmp)
    data = np.array(data)
    labels = np.float64(labels)
    labels.reshape((-1, num_classes))
    X_database = data
    Y_database = labels

    print("Load coco dataset complete...")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database

def get_imagenet():
    root = '/data/lyhe/BNN/DeepHash-pytorch-master/data/imagenet/'
    base_folder = 'train.txt'
    data = []
    labels = []
    num_classes = 100
    filename = os.path.join(root, base_folder)
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pos_tmp = lines.split()[0]
            pos_tmp = os.path.join(root, pos_tmp)
            label_tmp = lines.split()[1:]
            data.append(pos_tmp)
            labels.append(label_tmp)
    data = np.array(data)
    labels = np.float64(labels)
    labels.reshape((-1, num_classes))
    X_train = data
    Y_train = labels
    base_folder = 'test.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pos_tmp = lines.split()[0]
            pos_tmp = os.path.join(root, pos_tmp)
            label_tmp = lines.split()[1:]
            data.append(pos_tmp)
            labels.append(label_tmp)
    data = np.array(data)
    labels = np.float64(labels)
    labels.reshape((-1, num_classes))
    X_val = data
    Y_val = labels
    X_test = data
    Y_test = labels
    base_folder = 'database.txt'
    data = []
    labels = []
    filename = os.path.join(root, base_folder)
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pos_tmp = lines.split()[0]
            pos_tmp = os.path.join(root, pos_tmp)
            label_tmp = lines.split()[1:]
            data.append(pos_tmp)
            labels.append(label_tmp)
    data = np.array(data)
    labels = np.float64(labels)
    labels.reshape((-1, num_classes))
    X_database = data
    Y_database = labels

    print("Load imagenet dataset complete...")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database