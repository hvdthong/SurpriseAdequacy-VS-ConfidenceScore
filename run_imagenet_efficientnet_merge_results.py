import argparse
from utils import load_file, write_file
import os 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="imagenet")
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
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
        "--accuracy", "-accuracy", help="Accuracy of a model", action="store_true"
    )
    parser.add_argument(
        "--true_label", "-true_label", help="True Label", action="store_true"
    )
    parser.add_argument(
        "--pred_label", "-pred_label", help="Predicted Label", action="store_true"
    )
    parser.add_argument(
        "--adv_lsa", "-adv_lsa", help="Used Adversarial Examples for Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--adv_dsa", "-adv_dsa", help="Used Adversarial Examples for Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--adv_conf", "-adv_conf", help="Used Adversarial Examples for Confidence Score", action="store_true"
    )
    parser.add_argument("--model", "-model", help="Model for IMAGENET dataset", type=str, default="densenet201")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.lsa ^ args.dsa ^ args.conf ^ args.true_label ^ args.pred_label ^ args.adv_lsa ^ args.adv_dsa ^ args.adv_conf ^ args.accuracy, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    if args.d == 'imagenet':
        if args.pred_label:
            pred_label_all = list()
            for i in range(0, 50):
                pred_label = load_file('./metrics/%s_%s_pred_label_val_%i.txt' % (args.d, args.model, i))
                pred_label = [int(p) for p in pred_label]
                pred_label_all += pred_label
                os.remove('./metrics/%s_%s_pred_label_val_%i.txt' % (args.d, args.model, i))
            print(len(pred_label_all))
            write_file('./metrics/%s_%s_pred_label.txt' % (args.d, args.model), pred_label_all)

        if args.accuracy:
            pred_label_all, true_label_all = list(), list()
            for i in range(0, 50):
                pred_label = load_file('./metrics/%s_%s_pred_label_val_%i.txt' % (args.d, args.model, i))
                pred_label = [int(p) for p in pred_label]
                pred_label_all += pred_label

                true_label = load_file('./metrics/%s_%s_true_label_val_%i.txt' % (args.d, args.model, i))
                true_label = [int(p) for p in true_label]
                true_label_all += true_label
            print(len(pred_label_all), len(true_label_all))
            correct = len([p for p, l in zip(pred_label_all, true_label_all) if p == l])
            print('Accuracy of the IMAGENET dataset using model %s: %.4f' % (args.model, correct / len(pred_label_all)))

        if args.true_label:
            true_label_all = list()
            for i in range(0, 50):
                true_label = load_file('./metrics/%s_%s_true_label_val_%i.txt' % (args.d, args.model, i))
                true_label = [int(p) for p in true_label]
                true_label_all += true_label
                os.remove('./metrics/%s_%s_true_label_val_%i.txt' % (args.d, args.model, i))
            print(len(true_label_all))
            write_file('./metrics/%s_%s_true_label.txt' % (args.d, args.model), true_label_all)

        if args.conf:
            conf_all = list()
            for i in range(0, 50):
                conf = load_file('./metrics/%s_%s_conf_val_%i.txt' % (args.d, args.model, i))
                conf = [float(p) for p in conf]
                conf_all += conf
                os.remove('./metrics/%s_%s_conf_val_%i.txt' % (args.d, args.model, i))
            print(len(conf_all))
            write_file('./metrics/%s_%s_conf.txt' % (args.d, args.model), conf)