import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
import os


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def infog(msg):
    return Colors.OKGREEN + msg + Colors.ENDC


def info(msg):
    return Colors.OKBLUE + msg + Colors.ENDC


def warn(msg):
    return Colors.WARNING + msg + Colors.ENDC


def fail(msg):
    return Colors.FAIL + msg + Colors.ENDC


def compute_roc(probs_neg, probs_pos):
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


def compute_roc_auc(test_sa, adv_sa, split=1000):
    tr_test_sa = np.array(test_sa[:split])
    tr_adv_sa = np.array(adv_sa[:split])

    tr_values = np.concatenate(
        (tr_test_sa.reshape(-1, 1), tr_adv_sa.reshape(-1, 1)), axis=0
    )
    tr_labels = np.concatenate(
        (np.zeros_like(tr_test_sa), np.ones_like(tr_adv_sa)), axis=0
    )

    lr = LogisticRegressionCV(cv=5, n_jobs=-1).fit(tr_values, tr_labels)

    ts_test_sa = np.array(test_sa[split:])
    ts_adv_sa = np.array(adv_sa[split:])
    values = np.concatenate(
        (ts_test_sa.reshape(-1, 1), ts_adv_sa.reshape(-1, 1)), axis=0
    )
    labels = np.concatenate(
        (np.zeros_like(ts_test_sa), np.ones_like(ts_adv_sa)), axis=0
    )

    probs = lr.predict_proba(values)[:, 1]

    _, _, auc_score = compute_roc(
        probs_neg=probs[: (len(test_sa) - split)],
        probs_pos=probs[(len(test_sa) - split) :],
    )
    return auc_score


def write_file(path_file, data):
    split_path = path_file.split("/")
    path_ = split_path[:len(split_path) - 1]
    path_ = "/".join(path_)

    if not os.path.exists(path_):
        os.makedirs(path_)
    with open(path_file, 'w') as out_file:
        for line in data:
            # write line to output file
            out_file.write(str(line))
            out_file.write("\n")
        out_file.close()


def load_file(path_file):
    lines = list(open(path_file, "r").readlines())
    lines = [l.strip() for l in lines]
    return lines


def convert_list_number_to_float(data):
    return [float(d) for d in data]


def convert_list_string_to_True_False(data):
    return [True if d == 'True' else False for d in data]


def convert_predict_and_true_to_binary(predicted, true):
    # if instances are labeled correct, labeled them as 1 otherwise 0
    return [1 if p == t else 0 for p, t in zip(predicted, true)]


def use_for_confidnet_cifar10(data, random_number):
    import random
    random.seed(0)
    idx = [x for x in range(10000)]
    idx_random = random.sample(idx, random_number)
    for i in idx_random:
        if data[i] is True:
            data[i] = False    
    return data


from os import listdir
from os.path import isfile, join
def load_all_files(path_folder):  
    # load all files in a folder
    files = [f for f in listdir(path_folder) if isfile(join(path_folder, f))]
    return files

