import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from keras import utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

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

def load_img_imagenet(img_path):
    """Process the image of ImageNet data
    
    Args:
        img_path (string): Path of image
    Returns:        
        x (array): array of the image        
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_img_random_train(path_img, path_info):
    


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
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.lsa ^ args.dsa ^ args.conf ^ args.true_label ^ args.pred_label ^ args.adv_lsa ^ args.adv_dsa ^ args.adv_conf, "Select either 'lsa' or 'dsa' or etc."
    print(args)

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
        model = VGG16(weights='imagenet')
        model.summary()
        # exit()
        path_img_val = '../datasets/ilsvrc2012/images/val/'
        path_file_header = '../datasets/ilsvrc2012/images/val.txt'
        img_name, img_label = load_header_imagenet(load_file(path_file_header))
        print(len(img_name), len(img_label))
        print(img_name[0], img_label[0])

        path_img_val = '../datasets/ilsvrc2012/images/train/'
        path_file_header = '../datasets/ilsvrc2012/images/train.txt'
        img_name, img_label = load_header_imagenet(load_file(path_file_header))
        print(len(img_name), len(img_label))
        print(img_name[0], img_label[0])
        exit()
        for n, l in zip(img_name, img_label):
            img = load_img_imagenet(path_img_val + n)
            pred_img = model.predict(img)

            print(img.shape, l)
            print(pred_img.shape)
            print(np.amax(pred_img, axis=1))
            print(np.argmax(pred_img, axis=1))
            exit()
        print(len(img_name), len(img_label))
        print(img_name[0], img_label[0])
        img_path = 'elephant.jpg'
        exit()
        model = VGG16(weights='imagenet')
        model.summary()
        exit()

    if args.d == 'mnist' or args.d == 'cifar':
        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    if args.d == 'imagenet':


    if args.lsa:            
        test_lsa = fetch_lsa(model, x_train, x_test, "test", [args.layer], args)
        write_file(path_file='./metrics/{}_lsa_{}.txt'.format(args.d, args.layer), data=test_lsa)

    if args.dsa:
        test_dsa = fetch_dsa(model, x_train, x_test, "test", [args.layer], args)
        write_file(path_file='./metrics/{}_dsa_{}.txt'.format(args.d, args.layer), data=test_dsa)

    if args.conf:
        y_pred = model.predict(x_test)        
        test_conf = list(np.amax(y_pred, axis=1))
        write_file(path_file='./metrics/{}_conf.txt'.format(args.d), data=test_conf)

    if args.true_label:
        if args.d == 'mnist' or args.d == 'cifar':
            num_class = 10
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
