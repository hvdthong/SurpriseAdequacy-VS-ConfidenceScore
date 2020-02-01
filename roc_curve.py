import argparse
from utils import load_file, convert_list_number_to_float

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    print(args)

    predicted = load_file('./metrics/%s_pred_label.txt' % (args.d))
    true = load_file('./metrics/%s_true_label.txt' % (args.d))
    confidence = load_file('./metrics/%s_conf.txt' % (args.d))

    if args.d == 'mnist':
        lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_3.txt' % (args.d)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_3.txt' % (args.d)))
    
    if args.d == 'cifar':
        lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_11.txt' % (args.d)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_11.txt' % (args.d)))

    print(len(predicted), len(true), len(confidence), len(lsa), len(dsa))

