import argparse
from utils import load_file, convert_predict_and_true_to_binary, convert_list_number_to_float
import matplotlib.pyplot as plt


def box_plot_metrics(binary_predicted_true, score, args):
    """Draw the box plot for all the metrics (i.e., suprprise adequacy score, confidence score, etc.)

    Args:
        binary_predicted_true (list): different between the predicted and true instances (1 or 0)
        score (list): Set of instances for the metrics (i.e., suprprise adequacy score, confidence score, etc.).        
        args: Keyboard args.

    Returns:
        The box plot of the metric (i.e., suprprise adequacy score, confidence score, etc.)
    """
    correct, incorrect = list(), list()
    for b, s in zip(binary_predicted_true, score):
        if args.d == 'mnist' and args.lsa:
            if s <= 1500:
                if b == 1:
                    correct.append(s)
                else:
                    incorrect.append(s)
    print(len(correct), len(incorrect))
    
    if args.lsa:
        metric = 'lsa'
    if args.dsa:
        metric = 'dsa'
    if args.conf:
        metric = 'conf'

    data = [correct, incorrect]
    fig, ax = plt.subplots()
    ax.set_title('{} - {}'.format(metric, args.d))
    ax.boxplot(data)
    ax.set_xticklabels(['Correct', 'Incorrect'])
    plt.savefig('./results/{}_{}_box_plot.jpg'.format(args.d, metric))


if __name__ == '__main__':
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

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa ^ args.conf, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    predicted = load_file('./metrics/%s_pred_label.txt' % (args.d))
    true = load_file('./metrics/%s_true_label.txt' % (args.d))
    binary_predicted_true = convert_predict_and_true_to_binary(predicted=predicted, true=true)

    if args.d == 'mnist' and args.lsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_3.txt' % (args.d)))

    if args.d == 'mnist' and args.dsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_3.txt' % (args.d)))

    if args.conf:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_conf.txt' % (args.d)))

    box_plot_metrics(binary_predicted_true=binary_predicted_true, score=score_, args=args)

