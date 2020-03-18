import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from keras import utils
import random
import pickle

from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.preprocessing import image


random.seed(0)

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def load_header_imagenet(data):
    """Getting the header ImageNet data
    
    Args:
        data (list): List of imagenet information        
    Returns:        
        img_name (list of string): name of the image 
        label (list of int): label of the image 
    """
    img_name, label = list(), list()
    for d in data:
        img_name.append(d.strip().split()[0].strip())
        label.append(int(d.strip().split()[1].strip()))
    return img_name, label

def load_img_imagenet(img_path, args):
    """Process the image of ImageNet data
    
    Args:
        img_path (string): Path of image
    Returns:        
        x (array): array of the image        
    """
    if args.model == 'vgg16' or args.model == 'densenet201':
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if args.model == 'vgg16':
            from keras.applications.vgg16 import preprocess_input        
        elif args.model == 'densenet201':
            from keras.applications.densenet import preprocess_input
        x = preprocess_input(x)
        return x
    elif args.model == 'efficientnetb0' or args.model == 'efficientnetb7':         
        from efficientnet.keras import center_crop_and_resize, preprocess_input       
        from skimage.io import imread
        image = imread(img_path)
        image_size = args.image_size
        x = center_crop_and_resize(image, image_size=image_size)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        return x 


def numpy_append_advance(data):
    """Append all the elements of a list in a numpy array
    
    Args:
        data (list): List of array
    Returns:        
        new_data (array): Numpy array
    """
    new_data = None
    for i, d in enumerate(data):
        if i == 0:
            new_data = d
        else:
            new_data = np.append(new_data, d, axis=0)
    return new_data

def get_label_imagenet(name_folder, imagnet_info):    
    for i in imagnet_info:
        i_folder = i.split('/')[0]
        if name_folder == i_folder:
            return int(i.split()[1])

def load_imagenet_random_train(path_img, path_info, args):
    """Load IMAGENET dataset random training for calculating DSA or LSA
    
    Args:
        path_img (string): Folder of train dataset
        path_info (string): A file which includes the information of training data
        args: Keyboards arguments
    Returns:        
        x_train (array), y_train (array): Numpy array for input and label
    """
    imgnet_info = load_file(path_info)
    name_folders = [p.split('/')[0].strip() for p in imgnet_info]
    name_folders = list(sorted(set(name_folders)))
    
    if args.random_train == True:
        for i, n in enumerate(name_folders):
            random_name_file = sorted(random.sample(os.listdir(path_img + n), args.random_train_size))
            process_folder = numpy_append_advance([load_img_imagenet(path_img + n + '/' + r, args) for r in random_name_file])
            label_folder = get_label_imagenet(name_folder=n, imagnet_info=imgnet_info)
            label_folder = np.array([label_folder for i in range(args.random_train_size)])           
            print('Processing folder %i with have name: %s' % (i, n))
            pickle.dump((process_folder, label_folder), open('./dataset/imagenet/%s_%i_%s_.p' % (args.model, i, n), 'wb'))
        print('--------------------------------------------------')
        print('We finish processing the IMAGENET dataset')
        print('--------------------------------------------------')
        
        path_file = './dataset/imagenet/'
        x_random_train, y_random_train = list(), list()
        for i, n in enumerate(name_folders):
            x, y = pickle.load(open(path_file + args.model + '_' + str(i) + '_' + str(n) + '_.p', 'rb'))            
            x_random_train.append(x)
            y_random_train.append(y)     
        x_random_train = np.concatenate(x_random_train, axis=0)
        y_random_train = np.concatenate(y_random_train, axis=0)
        pickle.dump((x_random_train, y_random_train), open('./dataset/imagenet_train_%s.p' % (args.model), 'wb'), protocol=4)
        return x_random_train, y_random_train

    else:
        path_file = './dataset/imagenet_train_%s.p' % (args.model)
        x, y = pickle.load(open(path_file, 'rb'))
        return x, y
        
def load_imagenet_val(path_img, path_info, args):
    if args.val == True:
        img_name, img_label = load_header_imagenet(load_file(path_val_info))        
        x, y = list(), list()
        for n, l in zip(img_name, img_label):        
            x.append(load_img_imagenet(path_img + n, args))
            y.append(np.array([l]))
        x, y = np.concatenate(x), np.concatenate(y)                
        pickle.dump((x, y), open('./dataset/imagenet_val_%s.p' % (args.model), 'wb'), protocol=4)        
        return x, y
    else:
        path_file = './dataset/imagenet_val_%s.p' % (args.model)
        x, y = pickle.load(open(path_file, 'rb'))
        return x, y    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--conf", "-conf", help="Confidence Score", action="store_true"
    )    
    parser.add_argument(
        "--true_label", "-true_label", help="True Label", action="store_true"
    )
    parser.add_argument(
        "--pred_label", "-pred_label", help="Predicted Label", action="store_true"
    )
    parser.add_argument(
        "--adv_lsa", "-adv_lsa", help="Used Adversarial Examples for Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--adv_dsa", "-adv_dsa", help="Used Adversarial Examples for Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--adv_conf", "-adv_conf", help="Used Adversarial Examples for Confidence Score", action="store_true"
    )
    """We have five different attacks:
        + Fast Gradient Sign Method (fgsm)
        + Basic Iterative Method (bim-a, bim-b, or bim)
        + Jacobian-based Saliency Map Attack (jsma)
        + Carlini&Wagner (c+w)
    """
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
        # default=1e-10,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--layer",
        "-layer",
        help="Layer name",
        type=str,
    )
    parser.add_argument(
        "--random_train", "-random_train", help="random selected images for training (only for IMAGENET dataset)", action="store_true"
    )
    parser.add_argument("--random_train_size", "-random_train_size", help="Dataset", type=int, default=50)
    parser.add_argument(
        "--val", "-val", help="load validation dataset (only for IMAGENET dataset)", action="store_true"
    )
    parser.add_argument("--model", "-model", help="Model for IMAGENET dataset", type=str, default="densenet201")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.lsa ^ args.dsa ^ args.conf ^ args.true_label ^ args.pred_label ^ args.adv_lsa ^ args.adv_dsa ^ args.adv_conf, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    if args.d == 'imagenet':
        args.num_classes = 1000
        print('Loading training IMAGENET dataset -----------------------------')
        path_img_train = '../datasets/ilsvrc2012/images/train/'
        path_train_info = '../datasets/ilsvrc2012/images/train.txt'
        x_train, y_train = load_imagenet_random_train(path_img=path_img_train, path_info=path_train_info, args=args)

        print('Loading validation IMAGENET dataset -----------------------------')
        path_img_val = '../datasets/ilsvrc2012/images/val/'
        path_val_info = '../datasets/ilsvrc2012/images/val.txt'        
        x_test, y_test = load_imagenet_val(path_img=path_img_val, path_info=path_val_info, args=args)                
        print('Finish: Loading IMAGENET dataset -----------------------------')

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Load pre-trained model.
        model = load_model("./model/mnist_model_improvement-235-0.99.h5")
        model.summary()
    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        model = load_model("./model/cifar_model_improvement-491-0.88.h5")
        model.summary()
    elif args.d == 'imagenet':
        if args.model == 'vgg16':
            model = VGG16(weights='imagenet')
            model.summary()
        elif args.model == 'densenet201':
            model = DenseNet201(weights='imagenet')
            model.summary()


    if args.d == 'mnist' or args.d == 'cifar':
        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)


    if args.lsa:
        if args.d == 'mnist' or args.d == 'cifar':
            test_lsa = fetch_lsa(model, x_train, x_test, "test", [args.layer], args)
            write_file(path_file='./metrics/{}_lsa_{}.txt'.format(args.d, args.layer), data=test_lsa)
        elif args.d == 'imagenet':
            test_lsa = fetch_lsa(model, x_train, x_test, "test", [args.layer], args)
            write_file(path_file='./metrics/{}_lsa_{}_{}.txt'.format(args.d, args.model, args.layer), data=test_lsa)

    if args.dsa:
        if args.d == 'mnist' or args.d == 'cifar':
            test_dsa = fetch_dsa(model, x_train, x_test, "test", [args.layer], args)
            write_file(path_file='./metrics/{}_dsa_{}.txt'.format(args.d, args.layer), data=test_dsa)
        elif args.d == 'imagenet':
            test_dsa = fetch_dsa(model, x_train, x_test, "test", [args.layer], args)
            write_file(path_file='./metrics/{}_dsa_{}_{}.txt'.format(args.d, args.model, args.layer), data=test_dsa)

    if args.conf:
        if args.d == 'mnist' or args.d == 'cifar':
            y_pred = model.predict(x_test)
            test_conf = list(np.amax(y_pred, axis=1))
            write_file(path_file='./metrics/{}_conf.txt'.format(args.d), data=test_conf)
        elif args.d == 'imagenet':
            y_pred = model.predict(x_test)
            test_conf = list(np.amax(y_pred, axis=1))
            write_file(path_file='./metrics/{}_conf_{}.txt'.format(args.d, args.model), data=test_conf)
            
            # y_pred = np.argmax(y_pred, axis=1).tolist()
            # y_test = list(y_test)
            # correct = len([p for p, l in zip(y_pred, y_test) if p == l])
            # print('Accuracy of the IMAGENET dataset using model %s: %.4f' % (args.model, correct / len(y_pred)))

    if args.true_label:
        if args.d == 'mnist' or args.d == 'cifar':
            num_class = 10        
        elif args.d == 'imagenet':
            num_class = 1000
        else:
            print('Please input the number of classes')
            exit()
        y_test = utils.to_categorical(y_test, num_class)        
        y_test = np.argmax(y_test, axis=1)
        write_file(path_file='./metrics/{}_true_label.txt'.format(args.d), data=y_test)

    if args.pred_label:
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        write_file(path_file='./metrics/{}_pred_label.txt'.format(args.d), data=y_pred)

    if args.adv_lsa:
        x_adv = np.load('./adv/{}_{}.npy'.format(args.d, args.attack))        
        x_adv_lsa = fetch_lsa(model, x_train, x_adv, "adv_{}".format(args.attack), [args.layer], args)
        write_file(path_file='./metrics/{}_adv_lsa_{}_{}.txt'.format(args.d, args.attack, args.layer), data=x_adv_lsa)

    if args.adv_dsa:
        x_adv = np.load('./adv/{}_{}.npy'.format(args.d, args.attack))
        x_adv_dsa = fetch_dsa(model, x_train, x_adv, "adv_{}".format(args.attack), [args.layer], args)
        write_file(path_file='./metrics/{}_adv_dsa_{}_{}.txt'.format(args.d, args.attack, args.layer), data=x_adv_dsa)

    if args.adv_conf:
        x_adv = np.load('./adv/{}_{}.npy'.format(args.d, args.attack))
        x_adv_conf = model.predict(x_adv)
        x_adv_conf = list(np.amax(x_adv_conf, axis=1))
        write_file(path_file='./metrics/{}_adv_conf_{}.txt'.format(args.d, args.attack), data=x_adv_conf)
