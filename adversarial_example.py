import argparse
import tensorflow as tf
from keras import backend
import keras
from keras.datasets import mnist, cifar10

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
    print(args)

    # Force TensorFlow to use single thread to improve reproducibility
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    if keras.backend.image_data_format() != 'channels_last':
        raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)        

        model = load_model('./model/mnist_model_improvement-235-0.99.h5')
        classifier = KerasClassifier(model=model, clip_values=(0, 255), use_logits=False)

    if args.attack == 'fgsm':
        print('hello')
        