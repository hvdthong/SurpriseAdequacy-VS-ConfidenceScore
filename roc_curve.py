import argparse
from utils import load_file, convert_list_number_to_float
from sklearn.metrics import roc_curve, auc


def convert_predict_and_true_to_binary(predicted, true):
    # if instances are labeled correct, labeled them as 1 otherwise 0
    return [1 if p == t else 0 for p, t in zip(predicted, true)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    print(args)

    predicted = load_file('./metrics/%s_pred_label.txt' % (args.d))
    true = load_file('./metrics/%s_true_label.txt' % (args.d))
    confidence = convert_list_number_to_float(load_file('./metrics/%s_conf.txt' % (args.d)))

    if args.d == 'mnist':
        lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_3.txt' % (args.d)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_3.txt' % (args.d)))
    
    if args.d == 'cifar':
        lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_11.txt' % (args.d)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_11.txt' % (args.d)))
    
    binary_predicted_true = convert_predict_and_true_to_binary(predicted=predicted, true=true)

    fpr_conf, tpr_conf, _ = roc_curve(binary_predicted_true, confidence)
    roc_auc_conf = auc(fpr_conf, tpr_conf)
    
    ################################################################################################################
    # if we use the SA score, we should make it negative since it has the opposite meaning with the confidence score
    ################################################################################################################
    lsa = [-float(s) for s in lsa]  
    fpr_lsa, tpr_lsa, _ = roc_curve(binary_predicted_true, lsa)
    roc_auc_lsa = auc(fpr_lsa, tpr_lsa)

    ################################################################################################################
    dsa = [-float(s) for s in dsa]
    fpr_dsa, tpr_dsa, _ = roc_curve(binary_predicted_true, dsa)
    roc_auc_dsa = auc(fpr_dsa, tpr_dsa)
    ################################################################################################################

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_conf, tpr_conf, 'b', label = 'AUC_conf = %0.2f' % roc_auc_conf)
    plt.plot(fpr_lsa, tpr_lsa, 'c', label = 'AUC_lsa = %0.2f' % roc_auc_lsa)
    plt.plot(fpr_dsa, tpr_dsa, 'g', label = 'AUC_dsa = %0.2f' % roc_auc_dsa)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./results/%s_roc_curve.jpg' % (args.d))