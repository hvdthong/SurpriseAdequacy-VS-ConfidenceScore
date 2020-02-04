import argparse
import tensorflow as tf
from keras import backend
import keras
from keras.datasets import mnist, cifar10
from keras.models import load_model
from art.classifiers import KerasClassifier
import numpy as np

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == '__main__':
    """Generate the adversarial examples for mnist and cifar10 datasets
    Save them to the folder "adv" (adversarial examples)
    We have five different attacks:
        + Fast Gradient Sign Method (fgsm)
        + Basic Iterative Method (bim-a, bim-b)
        + Jacobian-based Saliency Map Attack (jsma)
        + Carlini&Wagner (c+w)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim-a", "bim-b", "jsma", "c+w"], "Attack we should used"
    print(args)

    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()        
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        x_test = x_test.astype("float32")
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        # load pre-trained model
        model = load_model('./model/mnist_model_improvement-235-0.99.h5')
        classifier = KerasClassifier(model=model, clip_values=(-0.5, 0.5), use_logits=False)

    if args.attack == 'fgsm':
        attack = FastGradientMethod(classifier=classifier, eps=0.3, batch_size=64)
        x_test_adv = attack.generate(x=x_test)

    if args.attack == '':
        print('hello')
    
    # Accuracy of our test data
    pred = np.argmax(classifier.predict(x_test), axis = 1)
    acc =  np.mean(np.equal(pred, y_test.reshape(-1)))

    print("The normal validation accuracy is: {}".format(acc))

    adv_pred = np.argmax(keras_model.predict(x_test_adv), axis = 1)
    adv_acc =  np.mean(np.equal(adv_pred, y_test))

    print("The adversarial validation accuracy is: {}".format(adv_acc))
    print(x_test_adv.shape)

