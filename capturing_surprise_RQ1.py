import argparse
from utils import convert_predict_and_true_to_binary, load_file, convert_list_number_to_float
import pandas as pd 
import matplotlib.pyplot as plt


def accuracy_surprise_inputs(df, x_axis_tickles):    
    """Calculate the accuracy of a list of images based on x_axis_tickles 
    Args: 
        df (dataframe: 'binary', 'score'): 
            + 'binary': different between predicted and true instances
            + 'score': metric (i.e., surprise adequacy score, confidence score, etc.)
        x_axis_tickles: a list of images (number of images)
    """
    binary_predicted_true = df['binary'].values.tolist()
    performance = list()    
    for t in x_axis_tickles:        
        performance.append(sum(binary_predicted_true[:t]) / t * 100)        
    return performance


def draw_surprise_inputs(binary_predicted_true, score, args):
    """Draw the plot line for capturing the surprise for all the metrics (i.e., suprprise adequacy score, confidence score, etc.)
    RQ1 in the surprise adequacy paper
    Args:
        binary_predicted_true (list): different between the predicted and true instances (1 or 0)
        score (list): Set of instances for the metrics (i.e., suprprise adequacy score, confidence score, etc.).        
        args: Keyboard args.

    Returns:
        The plot line (ascending and descending) of the metric (i.e., suprprise adequacy score, confidence score, etc.)
    """

    df = pd.DataFrame(list(zip(binary_predicted_true, score)), columns=['binary', 'score'])
    x_axis_tickles = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    if args.lsa:
        metric = 'lsa'
    if args.dsa:
        metric = 'dsa'
    if args.conf:
        metric = 'conf'
    if args.ts:
        metric = 'ts'
    
    if args.lsa or args.dsa or args.conf or args.ts:
        # draw a line of ascending and descending 
        per_ascending = accuracy_surprise_inputs(df=df.sort_values(by=['score'], ascending=True), x_axis_tickles=x_axis_tickles)
        per_descending = accuracy_surprise_inputs(df=df.sort_values(by=['score'], ascending=False), x_axis_tickles=x_axis_tickles)
        plt.plot(x_axis_tickles, per_ascending, 'ro-', label='Ascending %s' % metric)
        plt.plot(x_axis_tickles, per_descending, 'bs-', label='Descending %s' % metric)
        plt.legend()
        plt.savefig('./results/{}_{}_surprise_input.jpg'.format(args.d, metric))


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
    parser.add_argument(
        "--ts", "-ts", help="Trust Score", action="store_true"
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"    
    assert args.lsa ^ args.dsa ^ args.conf ^ args.ts, "Select either 'lsa' or 'dsa' or etc."    
    print(args)

    predicted = load_file('./metrics/%s_pred_label.txt' % (args.d))
    true = load_file('./metrics/%s_true_label.txt' % (args.d))
    binary_predicted_true = convert_predict_and_true_to_binary(predicted=predicted, true=true)

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

    if args.d == 'mnist' and args.ts:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_ts_activation_3.txt' % (args.d)))
    if args.d == 'cifar' and args.ts:
        score_ = convert_list_number_to_float(load_file('./metrics/%s_ts_activation_11.txt' % (args.d)))
    
    draw_surprise_inputs(binary_predicted_true=binary_predicted_true, score=score_, args=args)
