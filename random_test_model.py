import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from keras.utils import np_utils
from generate_sample_data import load_random_sample_data

CLIP_MIN = -0.5
CLIP_MAX = 0.5


def data_process(data):
    train, test = data
    x_train, y_train = train
    x_test, y_test = test

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    train = (x_train, y_train)
    test = (x_test, y_test)
    data = (train, test)
    return data

def test(args, ntime, data_test):
    if args.d == 'mnist':
        path_model = './random_sample_model/%s/%i/model_-75-.h5' % (args.d, ntime)
        print(path_model)
        exit()
        model = load_model(path_model)
        model.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--r", "-r", help="How many times we want to sample dataset", type=int, default=100)
    parser.add_argument("--s", "-s", help="Start times of random sampling", type=int, default=0)
    parser.add_argument("--e", "-e", help="End times of random sampling", type=int, default=100)

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
    assert args.lsa ^ args.dsa ^ args.conf ^ args.true_label ^ args.pred_label, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    save_path = '../2020_FSE_Empirical/%s' % args.d
    print(save_path)
    for t in range(args.s, args.e):
        data = load_random_sample_data(save_path=save_path, ntime=t)
        data = data_process(data)
        _, testing = data
        x_test, y_test = testing 
        test(args=args, ntime=t, data_test=(x_test, y_test))
        exit()