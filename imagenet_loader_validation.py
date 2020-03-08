from utils import load_file
from run import load_header_imagenet
import os
from shutil import copyfile

def get_label_name(path_imagenet_train_info):
    image_path, label = load_header_imagenet(load_file(path_imagenet_train_info))
    name_index_dict = {}
    for p, l in zip(image_path, label):
        if l not in name_index_dict.keys():
            name_index_dict[l] = p.split('/')[0]
    return name_index_dict

def create_validation_folder(name_index_imagenet, path_imagenet_val, path_imagenet_val_info):
    root_path = '../datasets/ilsvrc2012/images/'
    for k in name_index_imagenet.keys():
        if not os.path.exists(root_path + 'val_loader/' + name_index_imagenet[k]):
            os.makedirs(root_path + 'val_loader/' + name_index_imagenet[k])
    
    val_info = load_file(path_imagenet_val_info)
    for line in val_info:
        path_image, index_image_label = line.split()[0].strip(), int(line.split()[1])
        root_image_file = path_imagenet_val + path_image
        name_image_label = name_index_imagenet[index_image_label]
        dest_folder = root_path + 'val_loader/' + name_image_label + '/'        
        dest_image_name = name_image_label + '_' + str(int(path_image.split('_')[2].replace('.JPEG', '')))
        dest_file = dest_folder + dest_image_name + '.JPEG'        
        copyfile(root_image_file, dest_file)        


if __name__ == '__main__':
    path_imagenet_train = '../datasets/ilsvrc2012/images/train/'
    path_imagenet_train_info = '../datasets/ilsvrc2012/images/train.txt'    

    path_imagenet_val = '../datasets/ilsvrc2012/images/val/'
    path_imagenet_val_info = '../datasets/ilsvrc2012/images/val.txt'

    name_index_imagenet = get_label_name(path_imagenet_train_info)
    create_validation_folder(name_index_imagenet, path_imagenet_val, path_imagenet_val_info)