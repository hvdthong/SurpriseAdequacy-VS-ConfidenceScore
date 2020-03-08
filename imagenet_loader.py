from utils import load_file
import random
import argparse
import os 
from run import load_header_imagenet, load_img_imagenet
import numpy as np
import pickle

random.seed(0)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def process_batch(path_imagenet_folder, data):
    image_path, label = load_header_imagenet(data)
    x_batch = list()
    for p in image_path:
        x = load_img_imagenet(path_imagenet_folder + p)
        x = x.reshape(-1, 3, 224, 224)
        x_batch.append(x)
    x_batch = np.concatenate(x_batch)
    y_batch = np.asarray(label)    
    return (x_batch, y_batch)


def saving_batch_size_imagenet(args, path_imagenet_folder, path_imagnet_info):
    imagenet_info = load_file(path_imagnet_info)    
    random.shuffle(imagenet_info)
    batches = batch(imagenet_info, args.batch_size)
    
    if 'train' in path_imagnet_info:
        path_save = './dataset/IMAGENET/train/'
    elif 'val' in path_imagnet_info:
        path_save = './dataset/IMAGENET/val/'
    else:
        print('Wrong path information for IMAGENET')
        exit()
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for i, b in enumerate(batches):        
        i_batch = process_batch(path_imagenet_folder=path_imagenet_folder, data=b)
        i_batch_x, i_batch_y = i_batch
        print(i_batch_x)
        print(i_batch_x.shape, i_batch_y.shape)
        
        pickle.dump(i_batch, open(path_save + 'batch_%i.p' %i, 'wb'), protocol=4)
        exit()
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    # )
    # args = parser.parse_args()
    # print(args)

    # path_imagenet_train = '../datasets/ilsvrc2012/images/train/'
    # path_imagenet_train_info = '../datasets/ilsvrc2012/images/train.txt'

    # # saving_batch_size_imagenet(args=args, path_imagenet_folder=path_imagenet_train, path_imagnet_info=path_imagenet_train_info)

    # path_imagenet_val = '../datasets/ilsvrc2012/images/val/'
    # path_imagenet_val_info = '../datasets/ilsvrc2012/images/val.txt'

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torch

    traindir = '../datasets/ilsvrc2012/images/train/'    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=2, pin_memory=True)
    print(len(train_loader))
    for i, (images, target) in enumerate(train_loader):
        print(images)
        print(target)
        print(images.shape, target.shape)
        print(type(images), type(target))
        exit()

    valdir = '../datasets/ilsvrc2012/images/val_loader/'    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=True,
        num_workers=2, pin_memory=True)
    print(len(val_loader))
    for i, (images, target) in enumerate(val_loader):
        print(images)
        print(target)
        print(images.shape, target.shape)
        exit()
    
    # path_imagenet_val = '../datasets/ilsvrc2012/images/val/'
    # valdir = path_imagenet_val
    # valid_dataset = datasets.ImageNet(
    #                     valdir, 
    #                     'val', 
    #                     False,
    #                     transforms.Compose([
    #                         transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize,
    #                     ]))
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=2, pin_memory=True)
    # for i, (images, target) in enumerate(valid_loader):
    #     print(images.shape, target.shape)
    #     exit()