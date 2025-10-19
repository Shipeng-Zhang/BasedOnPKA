# import torchvision.transforms as T
import copy
import os.path
import os
from reid.utils.feature_tools import *
import lreid_dataset.datasets as datasets
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data import IterLoader
import numpy as np

def get_data(cam_filter, data_dir, height, width, batch_size, workers, num_instances, name,select_num=0):
    # root = osp.join(data_dir, name)
    root = data_dir
    
    cam_id = int(cam_filter.split("cam")[-1])
    cam_filter = cam_id
    dataset = datasets.create(name, cam_filter,root)
    '''select some persons for training'''
    train_img_count = len(dataset.train)
    num_classes = dataset.num_classes
    # if select_num > 0:
    #     train = []
    #     for instance in dataset.train:
    #         if instance[1] < select_num:
    #             # new_id=id_2_id[instance[1]]
    #             train.append((instance[0], instance[1], instance[2], instance[3]))  #img_path, pid, camid, domain-id

    #     dataset.train = train
    #     dataset.num_train_pids = select_num  # 更新训练集中身份的数量
    #     train_img_count = len(train)  # 更新训练集中图像的数量

   

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    


    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    
    #train_loader = DataLoader(
    #Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
    #batch_size=32, num_workers=workers, sampler=sampler,
    #shuffle=not rmgs_flag, pin_memory=True, drop_last=True)


    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=128, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,transform=test_transformer),
                             batch_size=128, num_workers=workers,shuffle=False, pin_memory=True, drop_last=False)
    return [dataset, num_classes, train_loader, test_loader, init_loader, name+'cam_'+str(cam_filter), train_img_count]

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def build_data_loaders(cfg, training_set, testing_only_set, toy_num=0):
    # Create data loaders
    data_dir = cfg.data_dir
    height, width = (256, 128)
    training_loaders = [get_data(cam_filter, data_dir, height, width, cfg.batch_size, cfg.workers,
                                 cfg.num_instances, name = 'market1501',select_num=-1) for cam_filter in training_set]
    return training_loaders, training_loaders
