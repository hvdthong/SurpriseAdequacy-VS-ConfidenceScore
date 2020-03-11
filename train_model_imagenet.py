from keras.preprocessing import image
import argparse
from utils import load_file
from keras.applications.vgg16 import VGG16
from run import load_header_imagenet
import numpy as np
from keras.applications.resnet import ResNet152
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge

def preprocessing_imagenet(img_path, args):
    """Process the image of ImageNet data
    
    Args:
        img_path (string): Path of image
    Returns:        
        x (array): array of the image        
    """
    if args.model == 'inceptionresnetv2':
        img = image.load_img(img_path, target_size=(299, 299))
    elif args.model == 'nasnetlarge':
        img = image.load_img(img_path, target_size=(331, 331))
    else:
        img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if args.model == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
    elif args.model == 'resnet152':
        from keras.applications.resnet import preprocess_input
    elif args.model == 'resnet152v2':
        from keras.applications.resnet_v2 import preprocess_input
    elif args.model == 'densenet201':
        from keras.applications.densenet import preprocess_input
    elif args.model == 'inceptionresnetv2':
        from keras.applications.inception_resnet_v2 import preprocess_input
    elif args.model == 'nasnetlarge':
        from keras.applications.nasnet import preprocess_input
    x = preprocess_input(x)
    return x


def evaluation(args):
    path_img_val = '../datasets/ilsvrc2012/images/val/'
    path_val_info = '../datasets/ilsvrc2012/images/val.txt'        

    if args.model == 'vgg16':
        model = VGG16(weights='imagenet')
        model.summary()
    elif args.model == 'resnet152':
        model = ResNet152(weights='imagenet')
        model.summary()
    elif args.model == 'resnet152v2':
        model = ResNet152V2(weights='imagenet')
        model.summary()
    elif args.model == 'inceptionresnetv2':
        model = InceptionResNetV2(weights='imagenet')
        model.summary()
    elif args.model == 'densenet201':
        model = DenseNet201(weights='imagenet')
        model.summary()
    elif args.model == 'nasnetlarge':
        model = NASNetLarge(weights='imagenet')
        model.summary()

    name, label = load_header_imagenet(load_file(path_val_info))    
    pred = list()
    for i, n in enumerate(name):
        x = preprocessing_imagenet(path_img_val + n, args)
        pred.append(np.argmax(model.predict(x), axis=1)[0])
        if i % 1000 == 0:
            print(n)
    
    correct = len([p for p, l in zip(pred, label) if p == l])
    print('Accuracy of the IMAGENET dataset using model %s: %.4f' % (args.model, correct / len(label)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="imagenet")
    parser.add_argument("--model", "-model", help="Model for Imagenet", type=str, default="vgg16")
    args = parser.parse_args()
    assert args.d in ['imagenet'], "IMGAENET Dataset"
    print(args)

    evaluation(args)

