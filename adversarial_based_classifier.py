import argparse
import numpy as np
from utils import load_file, convert_list_number_to_float
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


def generate_labeled_data(x_adv, x_test):
    """Generate labeled data for adversarial examples and test dataset
       1 if an example is adversarial; otherwise 0
    """
    y_adv = [1 for x in x_adv]  # label 1 if the instances are adversarial instances
    y_test = [0 for x in x_test]
    y = y_adv + y_test
    return np.array(y)


def roc_auc_classify(x, y, args):
    """Return the accuracy of classification algorithms
    Args:
        x (array): List of numpy array for training and testing (features)
        y (array): List of numpy array for training and testing (labeled)
        args: Keyboard args.

    Returns:
        accuracy (float): The performance of our classifier 
    """
    x_train, x_test = x
    y_train, y_test = y    
    if args.alg == 'lr':
        clf = LogisticRegression().fit(x_train, y_train)    
    fpr_conf, tpr_conf, _ = roc_curve(y_true=y_test, y_score=clf.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr_conf, tpr_conf)
    return roc_auc


def classify_adv_based_metrics(x_adv, x_test, args):
    """Calculate the performance of the adversarial examples based on different metrics
    Metrics (i.e, likelihood-based, distance-based, confidence score, etc.)
    
    Args:
        x_adv (list): List of metrics of adversarial examples 
        x_test (list): List of metrics of test data 
        args: Keyboard args.

    Returns:
        roc-auc (float): The performance of our classifier in term of roc curve
    """
    y = generate_labeled_data(x_adv=x_adv, x_test=x_test)
    x = np.array(x_adv + x_test)
    x = np.reshape(x, (x.shape[0], 1))    
    skf = StratifiedKFold(n_splits=args.n_fold, random_state=0, shuffle=True)
    for train_index, test_index in skf.split(x, y):
        x_test, x_train = x[train_index], x[test_index]  # using 90% percent of data for testing, only 10% for training
        y_test, y_train = y[train_index], y[test_index]
        
        roc_auc = round(roc_auc_classify(x=(x_train, x_test), y=(y_train, y_test), args=args), 4)
        if args.clf_dsa:
            print('ROC-AUC of dataset {} with attack {} for distance-based surprise adequacy (dsa): {}'.format(args.d, args.attack, roc_auc))
        if args.clf_conf:
            print('ROC-AUC of dataset {} with attack {} for confidence score: {}'.format(args.d, args.attack, roc_auc))
        if args.clf_ts:
            print('ROC-AUC of dataset {} with attack {} for trust score: {}'.format(args.d, args.attack, roc_auc))
        if args.clf_confidnet:
            print('ROC-AUC of dataset {} with attack {} for confidnet score: {}'.format(args.d, args.attack, roc_auc))
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument('--alg', '-alg', help="Algorithm Classification", type=str, default="lr")
    parser.add_argument('--n_fold', '-n_fold', help="Number of folds", type=int, default=10)
    parser.add_argument(
        "--clf_dsa", "-clf_dsa", help="Classification based on Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--clf_lsa", "-clf_lsa", help="Classification based on Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--clf_conf", "-clf_conf", help="Classification based on Confidence Score", action="store_true"
    )
    parser.add_argument(
        "--clf_ts", "-clf_ts", help="Classification based on Trust Score", action="store_true"
    )
    parser.add_argument(
        "--clf_confidnet", "-clf_confidnet", help="Classification based on Confidnet Score", action="store_true"
    )
    """We have five different attacks:
        + Fast Gradient Sign Method (fgsm)
        + Basic Iterative Method (bim-a, bim-b, or bim)
        + Jacobian-based Saliency Map Attack (jsma)
        + Carlini&Wagner (c+w)
    """
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    args = parser.parse_args()

    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.alg in ["lr"], "Algorithm Classification"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.clf_dsa ^ args.clf_lsa ^ args.clf_conf ^ args.clf_ts ^ args.clf_confidnet, "Select classification based on metrics (i.e., dsa, lsa, conf, etc.)"
    print(args)

    if args.clf_dsa:        
        if args.d == 'mnist':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_dsa_{}_activation_3.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_dsa_activation_3.txt'.format(args.d)))
        elif args.d == 'cifar':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_dsa_{}_activation_11.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_dsa_activation_11.txt'.format(args.d)))
        classify_adv_based_metrics(x_adv=x_adv, x_test=x_test, args=args)

    if args.clf_conf:
        if args.d == 'mnist' or args.d == 'cifar':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_conf_{}.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_conf.txt'.format(args.d)))        
        classify_adv_based_metrics(x_adv=x_adv, x_test=x_test, args=args)

    if args.clf_ts:        
        if args.d == 'mnist':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_ts_{}_activation_3.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_ts_activation_3.txt'.format(args.d)))
        elif args.d == 'cifar':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_ts_{}_activation_11.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_ts_activation_11.txt'.format(args.d)))
        classify_adv_based_metrics(x_adv=x_adv, x_test=x_test, args=args)

    if args.clf_confidnet:
        if args.d == 'mnist':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_confidnet_epoch_11_{}.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_confidnet_score_epoch_11.txt'.format(args.d)))
        elif args.d == 'cifar':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}10_adv_confidnet_epoch_162_{}.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}10_confidnet_score_epoch_162.txt'.format(args.d)))
        classify_adv_based_metrics(x_adv=x_adv, x_test=x_test, args=args)