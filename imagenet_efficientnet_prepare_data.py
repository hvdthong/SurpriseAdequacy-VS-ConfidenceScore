from utils import load_file, write_file
import random

def write_path_file_training_imagenet(path_img, path_info):
    imgnet_info = load_file(path_info)
    files = list()
    for i in imgnet_info:
        print(path_img + i)
        files.append(path_img + i)
    print(len(files))

    if 'train' in path_img:
        random.seed(0)
        random.shuffle(files)
        write_file('./dataset_imagenet/imagenet_training.txt', files)
    if 'val' in path_img:
        write_file('./dataset_imagenet/imagenet_val.txt', files)

if __name__ == '__main__':    
    path_img_train = '../datasets/ilsvrc2012/images/train/'
    path_train_info = '../datasets/ilsvrc2012/images/train.txt' 
    write_path_file_training_imagenet(path_img_train, path_train_info)

    # path_img_val = '../datasets/ilsvrc2012/images/val/'
    # path_val_info = '../datasets/ilsvrc2012/images/val.txt'
    # write_path_file_training_imagenet(path_img_val, path_val_info)