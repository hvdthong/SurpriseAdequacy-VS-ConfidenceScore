import argparse
from utils import load_file, convert_list_number_to_float, convert_predict_and_true_to_binary, convert_list_string_to_True_False, use_for_confidnet_cifar10
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--model", "-model", help="Use for imagenet dataset", type=str, default="efficientnetb7")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    print(args)

    if args.d == 'mnist' or args.d == 'cifar':
        predicted = load_file('./metrics/%s_pred_label.txt' % (args.d))
        true = load_file('./metrics/%s_true_label.txt' % (args.d))
        confidence = convert_list_number_to_float(load_file('./metrics/%s_conf.txt' % (args.d)))

    if args.d == 'imagenet':
        predicted = load_file('./metrics/%s_%s_pred_label.txt' % (args.d, args.model))
        true = load_file('./metrics/%s_%s_true_label.txt' % (args.d, args.model))
        confidence = convert_list_number_to_float(load_file('./metrics/%s_%s_conf.txt' % (args.d, args.model)))

    if args.d == 'mnist':
        lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_3.txt' % (args.d)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_3.txt' % (args.d)))
        ts = convert_list_number_to_float(load_file('./metrics/%s_ts_activation_3.txt' % (args.d)))
        
        # confidnet_accurate = convert_list_string_to_True_False(load_file('./metrics/%s_confidnet_accurate_epoch_11.txt' % (args.d)))
        # confidnet_score = convert_list_number_to_float(load_file('./metrics/%s_confidnet_score_epoch_11.txt' % (args.d)))

        confidnet_accurate = convert_list_string_to_True_False(load_file('./metrics/%s_confidnet_accurate.txt' % (args.d)))
        confidnet_score = convert_list_number_to_float(load_file('./metrics/%s_confidnet_score.txt' % (args.d)))
    
    if args.d == 'cifar':
        lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_activation_11.txt' % (args.d)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_activation_11.txt' % (args.d)))
        ts = convert_list_number_to_float(load_file('./metrics/%s_ts_activation_11.txt' % (args.d)))

        # confidnet_accurate = convert_list_string_to_True_False(load_file('./metrics/%s10_confidnet_accurate_epoch_162.txt' % (args.d)))
        # confidnet_accurate = use_for_confidnet_cifar10(confidnet_accurate, 65)
        # confidnet_score = convert_list_number_to_float(load_file('./metrics/%s10_confidnet_score_epoch_162.txt' % (args.d)))

        confidnet_accurate = convert_list_string_to_True_False(load_file('./metrics/%s_confidnet_accurate.txt' % (args.d)))
        confidnet_score = convert_list_number_to_float(load_file('./metrics/%s_confidnet_score.txt' % (args.d)))
    
    if args.d == 'imagenet':
        # lsa = convert_list_number_to_float(load_file('./metrics/%s_lsa_fc1.txt' % (args.d)))
        # dsa = convert_list_number_to_float(load_file('./metrics/%s_dsa_fc1.txt' % (args.d)))
        # ts = convert_list_number_to_float(load_file('./metrics/%s_ts_fc1.txt' % (args.d))) 

        lsa = convert_list_number_to_float(load_file('./metrics/%s_%s_lsa_avg_pool.txt' % (args.d, args.model)))
        dsa = convert_list_number_to_float(load_file('./metrics/%s_%s_dsa_avg_pool.txt' % (args.d, args.model)))
        ts = convert_list_number_to_float(load_file('./metrics/%s_%s_ts_avg_pool.txt' % (args.d, args.model)))
    
    binary_predicted_true = convert_predict_and_true_to_binary(predicted=predicted, true=true)   

    ################################################################################################################
    if args.d == 'mnist' or args.d == 'cifar':
        fpr_confidnet, tpr_confidnet, _ = roc_curve(confidnet_accurate, confidnet_score)
        roc_auc_confidnet = auc(fpr_confidnet, tpr_confidnet)
    ################################################################################################################

    ################################################################################################################
    fpr_conf, tpr_conf, _ = roc_curve(binary_predicted_true, confidence)
    roc_auc_conf = auc(fpr_conf, tpr_conf)
    ################################################################################################################
    
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

    ################################################################################################################
    if args.d == 'mnist' or args.d == 'cifar' or args.d == 'imagenet':
        fpr_ts, tpr_ts, _ = roc_curve(binary_predicted_true, ts)
        roc_auc_ts = auc(fpr_ts, tpr_ts)
    ################################################################################################################

    # method I: plt
    if args.d == 'mnist':
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr_dsa, tpr_dsa, 'b', label = 'AUC_conf = %0.2f' % roc_auc_dsa) 
        plt.plot(fpr_lsa, tpr_lsa, 'c', label = 'AUC_lsa = %0.2f' % roc_auc_lsa)
        plt.plot(fpr_conf, tpr_conf, 'g', label = 'AUC_dsa = %0.2f' % roc_auc_conf) 
        plt.plot(fpr_ts, tpr_ts, 'm', label = 'AUC_ts = %0.2f' % roc_auc_ts) 
        plt.plot(fpr_confidnet, tpr_confidnet, 'k', label = 'AUC_confidnet = %0.2f' % roc_auc_confidnet) 
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('./results/%s_roc_curve.jpg' % (args.d))

    if args.d == 'cifar':
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr_conf, tpr_conf, 'b', label = 'AUC_conf = %0.2f' % roc_auc_conf)
        plt.plot(fpr_lsa, tpr_lsa, 'c', label = 'AUC_lsa = %0.2f' % roc_auc_lsa)
        plt.plot(fpr_dsa, tpr_dsa, 'g', label = 'AUC_dsa = %0.2f' % roc_auc_dsa)
        plt.plot(fpr_ts, tpr_ts, 'm', label = 'AUC_ts = %0.2f' % roc_auc_ts) 
        plt.plot(fpr_confidnet, tpr_confidnet, 'k', label = 'AUC_confidnet = %0.2f' % roc_auc_confidnet) 
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('./results/%s_roc_curve.jpg' % (args.d))

    if args.d == 'imagenet':
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr_conf, tpr_conf, 'b', label = 'AUC_conf = %0.2f' % roc_auc_conf)
        plt.plot(fpr_lsa, tpr_lsa, 'c', label = 'AUC_lsa = %0.2f' % roc_auc_lsa)
        plt.plot(fpr_dsa, tpr_dsa, 'g', label = 'AUC_dsa = %0.2f' % roc_auc_dsa)
        plt.plot(fpr_ts, tpr_ts, 'm', label = 'AUC_ts = %0.2f' % roc_auc_ts) 
        # plt.plot(fpr_confidnet, tpr_confidnet, 'k', label = 'AUC_confidnet = %0.2f' % roc_auc_confidnet) 
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('./results/%s_roc_curve.jpg' % (args.d))