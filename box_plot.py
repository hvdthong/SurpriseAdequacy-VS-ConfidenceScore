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
        elif args.d == 'cifar' and args.lsa:
            if s <= 1500:
                if b == 1:
                    correct.append(s)
                else:
                    incorrect.append(s)
        elif args.d == 'imagenet' and args.lsa:
            if s <= 500:
                if b == 1:
                    correct.append(s)
                else:
                    incorrect.append(s)
        else:
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
    if args.d == 'openstack' or args.d == 'qt':
        plt.savefig('./results/defect_{}_{}_box_plot.jpg'.format(args.d, metric))
    else:
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
    assert args.d in ["mnist", "cifar", 'imagenet', 'openstack', 'qt'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa ^ args.conf, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    if args.d == 'imagenet':
        predicted = load_file('./metrics/%s_efficientnetb7_pred_label.txt' % (args.d))
        true = load_file('./metrics/%s_efficientnetb7_true_label.txt' % (args.d))
    elif args.d == 'mnist' or args.d == 'cifar':
        predicted = load_file('./metrics/%s_pred_label.txt' % (args.d))
        true = load_file('./metrics/%s_true_label.txt' % (args.d))
    elif args.d == 'openstack' or args.d == 'qt':
        predicted = load_file('./metrics/defect_%s_pred_label.txt' % (args.d))
        true = load_file('./metrics/defect_%s_true_label.txt' % (args.d))
    else:
        print('wrong dataset')
        exit()
    binary_predicted_true = convert_predict_and_true_to_binary(predicted=predicted, true=true)

    if (args.d == 'openstack' or args.d == 'qt') and args.lsa:
        score_ = convert_list_number_to_float(load_file('./metrics/defect_%s_lsa.txt' % (args.d)))
    if (args.d == 'openstack' or args.d == 'qt') and args.dsa:
        score_ = convert_list_number_to_float(load_file('./metrics/defect_%s_dsa.txt' % (args.d)))

    if args.d == 'imagenet' and args.lsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_efficientnetb7_lsa_avg_pool.txt' % (args.d)))

    if args.d == 'imagenet' and args.dsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_efficientnetb7_dsa_avg_pool.txt' % (args.d)))

    if args.d == 'mnist' and args.lsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_3.txt' % (args.d)))

    if args.d == 'mnist' and args.dsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_3.txt' % (args.d)))

    if args.d == 'cifar' and args.lsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_11.txt' % (args.d)))

    if args.d == 'cifar' and args.dsa:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_11.txt' % (args.d)))

    if args.conf:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_conf.txt' % (args.d)))

    box_plot_metrics(binary_predicted_true=binary_predicted_true, score=score_, args=args)

