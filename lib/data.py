# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os import path
import os


def get_dataset(dset_name, batch_size, n_worker, data_root='../../data'):
    cifar_tran_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    cifar_tran_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    print('=> Preparing data..')
    if dset_name == 'cifar10':
        transform_train = transforms.Compose(cifar_tran_train)
        transform_test = transforms.Compose(cifar_tran_test)
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=n_worker, pin_memory=True, sampler=None)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
    elif dset_name == 'imagenet':
        # get dir
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')

        # preprocessing
        input_size = 224
        imagenet_tran_train = [
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        imagenet_tran_test = [
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose(imagenet_tran_train)),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(imagenet_tran_test)),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)
        n_class = 1000

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class

pin_memory = False # True
def get_split_dataset(dset_name, batch_size, n_worker, val_size, data_root='../data',
                      use_real_val=False, shuffle=True, mode=False):
    '''
        split the train set into train / val for rl search
    '''
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:  # every time we use the same order for the split subset
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('=> Preparing data: {}...'.format(dset_name))
    if dset_name == 'cifar10':
        use_real_val = False
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        if use_real_val:  # split the actual val set
            valset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
            n_val = len(valset)
            assert val_size < n_val
            indices = list(range(n_val))
            np.random.shuffle(indices)
            _, val_idx = indices[val_size:], indices[:val_size]
            train_idx = list(range(len(trainset)))  # all train set for train
        else:  # split the train set
            valset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test) # TODO
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=pin_memory) 
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=pin_memory) 
        n_class = 10
    
    elif dset_name == 'aquatic_mammals':
        IMAGE_SIZE = 32
        transform_train = [
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
        ]
        transform_test = [
            transforms.Resize(int(IMAGE_SIZE/0.875)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
        ]
        trainset = cub200(f"/mxd_storage/maxiaodong/autoPrune/dataset/cifar-100/train/{dset_name}", train=True, transform=transforms.Compose(transform_train))
        valset = cub200(f"/mxd_storage/maxiaodong/autoPrune/dataset", train=True, transform=transforms.Compose(transform_test))
        n_train = len(trainset)
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        assert val_size < n_train
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=pin_memory)
        n_class = 200
    elif dset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])

        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        valset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform) # TODO 这里train=True什么意思
        n_train = len(trainset)
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        assert val_size < n_train
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                  num_workers=n_worker, pin_memory=True) # TODO  sampler option is mutually exclusive with shuffle
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
    elif dset_name == 'flower':
        means = [0.485, 0.456, 0.406]
        std_devs = [0.229, 0.224, 0.225]
        input_size = 224
        down_size = 256
        down_size1 = 32
        rotation = 30
        if mode:
            # 大图片模式 # mobilenet, mobilenetv2,  会用到
            data_transforms = {
                'train': transforms.Compose([transforms.RandomRotation(rotation),
                                            transforms.RandomResizedCrop(input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means,std_devs)]),
                'test': transforms.Compose([transforms.Resize(down_size),
                                                    transforms.CenterCrop(input_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means,std_devs)])
            }
        else:
            # 小图片模式 # vgg16，resnet18, resnet34, resnet50 会用到
            data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(rotation),
                                        transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize((down_size1, down_size1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,std_devs)]),
            'test': transforms.Compose([transforms.Resize((down_size1, down_size1)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means,std_devs)])
            }
        
        test_path = "test" if use_real_val else "train" 
        image_datasets = {
            'train': datasets.ImageFolder('./dataset/flower_data/train/', 
                                        transform=data_transforms['train']),
            # 'valid': datasets.ImageFolder('./dataset/flower_data/train/', 
            #                             transform=data_transforms['train']),
            'test': datasets.ImageFolder(f'./dataset/flower_data/{test_path}/', 
                                        transform=data_transforms['test'])
            }

        trainset = image_datasets['train']
        
        if use_real_val:
            valset = image_datasets["test"]
            n_val = len(valset)
            # assert val_size < n_val
            indices = list(range(n_val))
            np.random.shuffle(indices)
            val_size = n_val
            print("train_size", len(trainset))
            print("val_size", val_size)
            _, val_idx = indices[val_size:], indices[:val_size]
            train_idx = list(range(len(trainset)))  # all train set for train
        else:  # split the train set
            valset = image_datasets["train"]
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        # Define dataloaders
        dataloaders = {
            'train': torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False),
            # 'valid': torch.utils.data.DataLoader(image_datasets['valid'], 
            #                                     batch_size=batch_size, shuffle=False, sampler=val_sampler,num_workers=n_worker, pin_memory=True), 
            'test': torch.utils.data.DataLoader(valset, 
                                                batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                   num_workers=n_worker, pin_memory=False)
            }
            
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]
        n_class = 102
    elif dset_name == 'flower1':
        means = [0.485, 0.456, 0.406]
        std_devs = [0.229, 0.224, 0.225]
        input_size = 224
        down_size = 256
        down_size1 = 32
        rotation = 30
        mode = False 
        if mode:
            # 大图片模式 # mobilenet, mobilenetv2,  会用到
            data_transforms = {
                'train': transforms.Compose([transforms.RandomRotation(rotation),
                                            transforms.RandomResizedCrop(input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means,std_devs)]),
                'test': transforms.Compose([transforms.Resize(down_size),
                                                    transforms.CenterCrop(input_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means,std_devs)])
                }
        else:
            # 小图片模式 # vgg16，resnet18, resnet34, resnet50 会用到
            data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(rotation),
                                        transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize((down_size1, down_size1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,std_devs)]),
            'test': transforms.Compose([transforms.Resize((down_size1, down_size1)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means,std_devs)])
            }

        # 使用假验证集
        image_datasets = {
            'train': datasets.ImageFolder('./dataset/flower1/train/', 
                                        transform=data_transforms['train']),
            'test': datasets.ImageFolder('./dataset/flower1/train/', 
                                        transform=data_transforms['train'])
            }
        n_train = len(image_datasets['train'])
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        assert val_size < n_train
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        # Define dataloaders
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False),
            'test': torch.utils.data.DataLoader(image_datasets['test'], 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False)
            }
            
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]
        n_class = 34
    elif dset_name == 'flower2':
        means = [0.485, 0.456, 0.406]
        std_devs = [0.229, 0.224, 0.225]
        input_size = 224
        down_size = 256
        down_size1 = 32
        rotation = 30
        mode = False 
        if mode:
            # 大图片模式 # mobilenet, mobilenetv2,  会用到
            data_transforms = {
                'train': transforms.Compose([transforms.RandomRotation(rotation),
                                            transforms.RandomResizedCrop(input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means,std_devs)]),
                'test': transforms.Compose([transforms.Resize(down_size),
                                                    transforms.CenterCrop(input_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means,std_devs)])
                }
        else:
            # 小图片模式 # vgg16，resnet18, resnet34, resnet50 会用到
            data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(rotation),
                                        transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize((down_size1, down_size1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,std_devs)]),
            'test': transforms.Compose([transforms.Resize((down_size1, down_size1)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means,std_devs)])
            }

        # 使用假验证集
        image_datasets = {
            'train': datasets.ImageFolder('./dataset/flower2/train/', 
                                        transform=data_transforms['train']),
            'test': datasets.ImageFolder('./dataset/flower2/train/', 
                                        transform=data_transforms['train'])
            }
        n_train = len(image_datasets['train'])
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        # print(val_size)
        # print(n_train)
        assert val_size < n_train
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        # Define dataloaders
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False),
            'test': torch.utils.data.DataLoader(image_datasets['test'], 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False)
            }
            
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]
        n_class = 34
    elif dset_name == 'flower3':
        means = [0.485, 0.456, 0.406]
        std_devs = [0.229, 0.224, 0.225]
        input_size = 224
        down_size = 256
        down_size1 = 32
        rotation = 30
        mode = False 
        if mode:
            # 大图片模式 # mobilenet, mobilenetv2,  会用到
            data_transforms = {
                'train': transforms.Compose([transforms.RandomRotation(rotation),
                                            transforms.RandomResizedCrop(input_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means,std_devs)]),
                'test': transforms.Compose([transforms.Resize(down_size),
                                                    transforms.CenterCrop(input_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(means,std_devs)])
                }
        else:
            # 小图片模式 # vgg16，resnet18, resnet34, resnet50 会用到
            data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(rotation),
                                        transforms.RandomResizedCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize((down_size1, down_size1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means,std_devs)]),
            'test': transforms.Compose([transforms.Resize((down_size1, down_size1)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means,std_devs)])
            }

        # 使用假验证集
        image_datasets = {
            'train': datasets.ImageFolder('./dataset/flower3/train/', 
                                        transform=data_transforms['train']),
            'test': datasets.ImageFolder('./dataset/flower3/train/', 
                                        transform=data_transforms['train'])
            }
        n_train = len(image_datasets['train'])
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        assert val_size < n_train
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        # Define dataloaders
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False),
            'test': torch.utils.data.DataLoader(image_datasets['test'], 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False)
            }
            
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]
        n_class = 34
    elif dset_name == 'flower5':
        input_size = 224
        down_size1 = 32

        # 大图片模式 mobilenetv2
        data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(input_size),       # 随机裁剪，再缩放成 224×224
                                     transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        "val": transforms.Compose([transforms.Resize((input_size, input_size)),  
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

        # 小图片模式 
        # data_transform = {
        #     "train": transforms.Compose([transforms.RandomResizedCrop(input_size), # 随机裁剪，再缩放成 224×224
        #                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
        #                                 transforms.Resize((down_size1, down_size1)),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        #     "val": transforms.Compose([transforms.Resize((down_size1, down_size1)),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

        # 获取图像数据集的路径
        data_root = os.path.abspath(os.path.join(os.getcwd()))  		# get data root path 返回上上层目录
        image_path = data_root + "/dataset/flower5/"  				 		# flower data_set path

        # 导入训练集并进行预处理
        train_dataset = datasets.ImageFolder(root=image_path + "/train",		
                                            transform=data_transform["train"])

        # 导入验证集并进行预处理
        val_dataset = datasets.ImageFolder(root=image_path + "/train",
                                            transform=data_transform["train"])

        n_train = len(train_dataset)
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        assert val_size < n_train
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        # Define dataloaders
        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False),
            'test': torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=False)
            }
            
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]
        n_class = 5
    elif dset_name == 'imagenet':
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input_size = 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(train_dir, train_transform)
        if use_real_val:
            valset = datasets.ImageFolder(val_dir, test_transform)
            n_val = len(valset)
            assert val_size < n_val
            indices = list(range(n_val))
            np.random.shuffle(indices)
            _, val_idx = indices[val_size:], indices[:val_size]
            train_idx = list(range(len(trainset)))  # all trainset
        else:
            valset = datasets.ImageFolder(train_dir, test_transform)
            n_train = len(trainset)
            indices = list(range(n_train))
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)

        n_class = 1000
    elif 'cifar100' in dset_name:
        names = dset_name.split("_", 1) # num 是分割次数，而不是分割后的元素个数
        image_size= 32
        padding= 4 
        transform_train = transforms.Compose([
            transforms.RandomCrop(image_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.RandomCrop(image_size, padding=padding), # 为了解决 torchvision.models.resnet18不能输入32图片
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 获取图像数据集的路径
        # data_root = os.path.abspath(os.path.join(os.getcwd()))  		# get data root path 返回上上层目录
        train_image_path = data_root + f"/train/{names[1]}"
        test_image_path = data_root + f"/test/{names[1]}"

        # 导入训练集并进行预处理
        train_dataset = datasets.ImageFolder(root=train_image_path, transform=transform_train)
        # test_dataset = datasets.ImageFolder(root=test_image_path, transform=transform_test)

        n_train = len(train_dataset)
        indices = list(range(n_train))
        # now shuffle the indices
        np.random.shuffle(indices)
        assert val_size < n_train, f"val_size: {val_size}; train_size: {n_train};"
        train_idx, val_idx = indices[val_size:], indices[:val_size] # list(range(n_train)) # 

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)
        # 按batch_size分批次加载训练集
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=pin_memory)
        n_class = 5
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
    

import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile
class cub200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(cub200, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform

        self.processed = "CUB_200_PROCESSED"

        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, self.processed + '/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, self.processed + '/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _check_processed(self):
        assert os.path.isdir(self.root) == True
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True
        return (os.path.isfile(os.path.join(self.root, self.processed + '/train.pkl')) and
                os.path.isfile(os.path.join(self.root, self.processed + '/test.pkl')))

    def _extract(self):
        processed_data_path = os.path.join(self.root, self.processed)
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: '+image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt%1000 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, self.processed + '/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, self.processed + '/test.pkl'), 'wb'))