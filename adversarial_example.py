import argparse
import tensorflow as tf
from keras import backend
import keras
from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.applications.vgg16 import VGG16

# Using https://github.com/IBM/adversarial-robustness-toolbox to create adversarial examples
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, CarliniL2Method, ProjectedGradientDescent
import numpy as np
from run import load_imagenet_val
from keras.applications.densenet import DenseNet201

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    """Generate the adversarial examples for mnist and cifar10 datasets
    Save them to the folder "adv" (adversarial examples)
    We have five different attacks:
        + Fast Gradient Sign Method (fgsm)
        + Basic Iterative Method (bim-a, bim-b, or bim)
        + Jacobian-based Saliency Map Attack (jsma)
        + Carlini&Wagner (c+w)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    parser.add_argument("--model", "-model", help="Model used for IMAGENET dataset", type=str, default="densenet201")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim-a", "bim-b", "bim", "jsma", "c+w"], "Attack we should used"
    print(args)

    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()        
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # load pre-trained model
        model = load_model('./model/mnist_model_improvement-235-0.99.h5')
        classifier = KerasClassifier(model=model, clip_values=(-0.5, 0.5), use_logits=False)

    if args.d == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # load pre-trained model
        model = load_model("./model/cifar_model_improvement-491-0.88.h5")
        classifier = KerasClassifier(model=model, clip_values=(-0.5, 0.5), use_logits=False)

    if args.d == 'imagenet':
        args.num_classes = 1000
        args.val = False

        path_img_val = '../datasets/ilsvrc2012/images/val/'
        path_val_info = '../datasets/ilsvrc2012/images/val.txt'        
        print('Loading validation dataset for IMAGENET----------------')
        x_test, y_test = load_imagenet_val(path_img=path_img_val, path_info=path_val_info, args=args)       
        print('Done ----------------')

        if args.model == 'vgg16':
            model = VGG16(weights='imagenet')
            classifier = KerasClassifier(model=model, clip_values=(-0.5, 0.5), use_logits=False)
        elif args.model == 'densenet201':            
            model = DenseNet201(weights='imagenet')            
            classifier = KerasClassifier(model=model, use_logits=False)


    if args.attack == 'fgsm':
        attack = FastGradientMethod(classifier=classifier, eps=0.6, eps_step=0.6, batch_size=64)

    if args.attack == 'bim':
        if args.d == 'imagenet':
            attack = BasicIterativeMethod(classifier=classifier, eps=0.6, batch_size=64, max_iter=25)
        else:    
            attack = BasicIterativeMethod(classifier=classifier, eps=0.6, batch_size=64)

    if args.attack == 'jsma':
        attack = SaliencyMapMethod(classifier=classifier, batch_size=64)

    if args.attack == 'c+w':
        attack = CarliniL2Method(classifier=classifier, batch_size=64)
    
    # generating adversarial of the testing dataset and save it to the folder './adv'
    if args.d == 'mnist' or args.d == 'cifar':
        x_adv = attack.generate(x=x_test)
        np.save('./adv/{}_{}.npy'.format(args.d, args.attack), x_adv)
    if args.d == 'imagenet':
        x_adv = attack.generate(x=x_test)
        np.save('./adv/{}_{}_{}.npy'.format(args.d, args.attack, args.model), x_adv)

    # accuracy of our test data
    pred = np.argmax(classifier.predict(x_test), axis=1)
    acc =  np.mean(np.equal(pred, y_test.reshape(-1)))
    print("The normal validation accuracy is: {}".format(acc))

    # accuracy of our adversarial examples generated by the test data    
    adv_pred = np.argmax(classifier.predict(x_adv), axis=1)    
    print(x_adv.shape)
    adv_acc =  np.mean(np.equal(adv_pred, y_test.reshape(-1)))
    print("The adversarial validation accuracy is: {}".format(adv_acc))

