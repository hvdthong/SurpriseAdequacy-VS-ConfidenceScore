import argparse
import pickle
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, CarliniL2Method
import efficientnet.keras as efn 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    parser.add_argument("--model", "-model", help="Model used for IMAGENET dataset", type=str, default="efficientnetb7")
    parser.add_argument(
        "--val_start", "-val_start", help="Start validation index (only for IMAGENET dataset)", type=int, default=0
    )
    parser.add_argument(
        "--val_end", "-val_end", help="End validation index (only for IMAGENET dataset)", type=int, default=50
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=1
    )
    args = parser.parse_args()
    assert args.d in ['imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim-a", "bim-b", "bim", "jsma", "c+w"], "Attack we should used"
    print(args)

    if args.d == 'imagenet':        
        print('Generate the adversarial example of dataset %s using model %s with the attack %s' % (args.d, args.model, args.attack))
        for i in range(args.val_start, args.val_end):
            x_test, y_test = pickle.load(open('./dataset_imagenet/%s_%s_val_%i.p' % (args.d, args.model, int(i)), 'rb'))
            print(i, x_test.shape, y_test.shape)

            if args.model == 'efficientnetb7':
                model = efn.EfficientNetB7(weights='imagenet')  # only use without modifying batch size (default: 1)
                classifier = KerasClassifier(model=model, use_logits=False)

            if args.attack == 'fgsm':
                attack = FastGradientMethod(classifier=classifier, eps=0.6, eps_step=0.6)  
            print('Generating adversarial examples----------------')
            x_adv = attack.generate(x=x_test)
            np.save('./adv/{}_{}_{}_val_%i.npy'.format(args.d, args.model, args.attack, i), x_adv)