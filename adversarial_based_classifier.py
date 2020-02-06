import argparse
import numpy as np
from utils import load_file, convert_list_number_to_float


def generate_labeled_data(x_adv, x_test):
    """Generate labeled data for adversarial examples and test dataset
       1 if an example is adversarial; otherwise 0
    """

def classify_adv_based_metrics(x_adv, x_test, args):
    """Calculate the performance of the adversarial examples based on different metrics
    Metrics (i.e, likelihood-based, distance-based, confidence score, etc.)
    
    Args:
        x_adv (list): List of metrics of adversarial examples 
        x_test (list): List of metrics of test data 
        args: Keyboard args.

    Returns:
        accuracy (float): The performance of our classifier 
    """
    print('hello')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument('--alg', '-alg', help="Algorithm Classification", type=str, default="lr")
    parser.add_argument('--n_fold', '-n_fold', help="Number of folds", type=int, default=10)
    parser.add_argument(
        "--clf_dsa", "-clf_dsa", help="Classification based on Distance-based Surprise Adequacy", action="store_true"
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
    assert args.clf_dsa, "Select classification based on metrics"
    print(args)

    if args.clf_dsa:        
        if args.d == 'mnist':
            x_adv = convert_list_number_to_float(load_file('./metrics/{}_adv_dsa_{}_activation_3.txt'.format(args.d, args.attack)))
            x_test = convert_list_number_to_float(load_file('./metrics/{}_dsa_activation_3.txt'.format(args.d)))
            classify_adv_based_metrics(x_adv=x_adv, x_test=x_test, args=args)