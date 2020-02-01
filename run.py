import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from keras import utils

CLIP_MIN = -0.5
CLIP_MAX = 0.5

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
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa ^ args.conf ^ args.true_label, "Select either 'lsa' or 'dsa'"
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

        model = load_model("./model/cifar_model_improvement-250-0.86.h5")
        model.summary()        

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

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

        