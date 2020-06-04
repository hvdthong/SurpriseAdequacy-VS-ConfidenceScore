import argparse
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
import numpy as np
from trust_score_example import trustscore
from keras.utils import np_utils
from utils import write_file
from run import load_imagenet_random_train, load_imagenet_val
from keras.applications.vgg16 import VGG16

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def get_attention_layer(x, model, args):
    """Getting the neurons of a particular layer (i.e., features)
    
    Args:
        x (array): Dataset for getting the features 
        model (deep learning model): Deep learning model
        args: Keyboard args.
    Returns:
        ats_layer (array): The features of the particular given the input x and model
    """
    layer_names = [args.layer] 
    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )
    if len(layer_names) == 1:
        ats_layer = np.array(temp_model.predict(x, batch_size=args.batch_size, verbose=1))
    else:
        print('You can only input one layer')
    return ats_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--ts", "-ts", help="Trust Score for Test Datasets (i.e., MNIST or CIFAR)", action="store_true"
    )
    parser.add_argument(
        "--adv_ts", "-adv_ts", help="Trust Score for Adversarial Examples", action="store_true"
    )
    parser.add_argument(
        "--layer",
        "-layer",
        help="Layer name",
        type=str,
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--random_train", "-random_train", help="random selected images for training (only for IMAGENET dataset)", action="store_true", default=False
    )
    parser.add_argument(
        "--val", "-val", help="load validation dataset (only for IMAGENET dataset)", action="store_true", default=False
    )
    """We have five different attacks:
        + Fast Gradient Sign Method (fgsm)
        + Basic Iterative Method (bim-a, bim-b, or bim)
        + Jacobian-based Saliency Map Attack (jsma)
        + Carlini&Wagner (c+w)
    """
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.ts ^ args.adv_ts, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    if args.ts:
        if args.d == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

            # Load pre-trained model.
            model = load_model("./model/mnist_model_improvement-235-0.99.h5")            
        elif args.d == 'cifar':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train, y_test = y_train.flatten(), y_test.flatten()

            # Load pre-trained model.
            model = load_model("./model/cifar_model_improvement-491-0.88.h5")
        
        elif args.d == 'imagenet':
            print('Loading IMAGENET dataset -----------------------------')
            path_img_train = '../datasets/ilsvrc2012/images/train/'
            path_train_info = '../datasets/ilsvrc2012/images/train.txt'
            x_train, y_train = load_imagenet_random_train(path_img=path_img_train, path_info=path_train_info, args=args)

            path_img_val = '../datasets/ilsvrc2012/images/val/'
            path_val_info = '../datasets/ilsvrc2012/images/val.txt'        
            x_test, y_test = load_imagenet_val(path_img=path_img_val, path_info=path_val_info, args=args)                
            print('Finish: Loading IMAGENET dataset -----------------------------')

        if args.d == 'mnist' or args.d == 'cifar':
            model.summary()
            
            x_train = x_train.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
            x_test = x_test.astype("float32")
            x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

            y_pred = np.argmax(model.predict(x_test), axis=1)

            x_train_ats_layer = get_attention_layer(x=x_train, model=model, args=args)
            x_test_ats_layer = get_attention_layer(x=x_test, model=model, args=args)        

            trust_model = trustscore.TrustScore()
            trust_model.fit(x_train_ats_layer, y_train)

            trust_score = trust_model.get_score(x_test_ats_layer, y_pred).tolist()        
            write_file(path_file='./metrics/{}_ts_{}.txt'.format(args.d, args.layer), data=trust_score)

        if args.d == 'imagenet':
            model = VGG16(weights='imagenet')
            model.summary()

            y_pred = np.argmax(model.predict(x_test), axis=1)

            x_train_ats_layer = get_attention_layer(x=x_train, model=model, args=args)
            x_test_ats_layer = get_attention_layer(x=x_test, model=model, args=args)        

            trust_model = trustscore.TrustScore()
            trust_model.fit(x_train_ats_layer, y_train)

            trust_score = trust_model.get_score(x_test_ats_layer, y_pred).tolist()        
            write_file(path_file='./metrics/{}_ts_{}.txt'.format(args.d, args.layer), data=trust_score)


    elif args.adv_ts: 
        if args.d == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

            # Load pre-trained model.
            model = load_model("./model/mnist_model_improvement-235-0.99.h5")

        elif args.d == 'cifar':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            model = load_model("./model/cifar_model_improvement-491-0.88.h5")
            y_train, y_test = y_train.flatten(), y_test.flatten()

        model.summary()

        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_adv = np.load('./adv/%s_%s.npy' % (args.d, args.attack))

        y_pred = np.argmax(model.predict(x_adv), axis=1)

        x_train_ats_layer = get_attention_layer(x=x_train, model=model, args=args)
        x_adv_ats_layer = get_attention_layer(x=x_adv, model=model, args=args)        

        trust_model = trustscore.TrustScore()
        trust_model.fit(x_train_ats_layer, y_train)

        trust_score = trust_model.get_score(x_adv_ats_layer, y_pred).tolist()
        write_file(path_file='./metrics/{}_adv_ts_{}_{}.txt'.format(args.d, args.attack, args.layer), data=trust_score)
        