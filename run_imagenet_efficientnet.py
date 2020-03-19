import efficientnet.keras as efn 
import argparse
from utils import load_file, write_file
from run import load_img_imagenet, get_label_imagenet, load_header_imagenet
import os 
import numpy as np 
import random 
from skimage.io import imread
import keras 
import pickle
from sa import fetch_lsa_imagenet, _get_kdes
from keras.models import Model

def load_imagenet_random_train(path_img, path_info, args):
    """Load IMAGENET dataset random training for calculating DSA or LSA
    
    Args:
        path_img (string): Folder of train dataset
        path_info (string): A file which includes the information of training data
        args: Keyboards arguments
    Returns:        
        x_train (array), y_train (array): Numpy array for input and label
    """
    imgnet_info = load_file(path_info)
    name_folders = [p.split('/')[0].strip() for p in imgnet_info]
    name_folders = list(sorted(set(name_folders)))    
    
    for m in range(10, args.random_train_num):
        random.seed(m)
        if os.path.exists('./dataset/%s_%s_random_train_%i.p' % (args.d, args.model, m)):                
            print('File exists in your directory')
            continue
        else:                
            random_train_file, x_random_train, y_random_train = list(), list(), list()
            for i, n in enumerate(name_folders):
                random_name_file = sorted(random.sample(os.listdir(path_img + n), args.random_train_size))                
                process_folder = [load_img_imagenet(path_img + n + '/' + r, args) for r in random_name_file]
                process_folder = [p for p in process_folder if len(p.shape) == 4]
                if len(process_folder) > 0:
                    process_folder = np.concatenate(process_folder, axis=0)
                    label_folder = get_label_imagenet(name_folder=n, imagnet_info=imgnet_info)                
                    label_folder = np.array([label_folder for i in range(args.random_train_size)])                                
                    
                    x_random_train.append(process_folder)
                    y_random_train.append(label_folder)                    
                    print('Random training %i-th of the folder %i-th which has name %s' % (m, i, n))
                else:
                    continue
            x_random_train = np.concatenate(x_random_train, axis=0)
            y_random_train = np.concatenate(y_random_train, axis=0)                
            pickle.dump((x_random_train, y_random_train), open('./dataset/%s_%s_random_train_%i.p' % (args.d, args.model, m), 'wb'), protocol=4)        
    print('Now you can load the training dataset')
    exit()

def load_imagenet_val(path_img, path_info, args):
    img_name, img_label = load_header_imagenet(load_file(path_val_info))

    chunks_name = [img_name[x:x+args.val_size] for x in range(0, len(img_name), args.val_size)]
    chunks_label = [img_label[x:x+args.val_size] for x in range(0, len(img_label), args.val_size)]

    for i, (c_n, c_l) in enumerate(zip(chunks_name, chunks_label)):
        if os.path.exists('./dataset/%s_%s_val_%i.p' % (args.d, args.model, i)):                
            print('File exists in your directory')
            continue
        else:        
            x_val, y_val = list(), list()
            for n, l in zip(c_n, c_l):                
                img = load_img_imagenet(path_img + n, args)
                if len(img.shape) == 4:
                    x_val.append(img)
                    y_val.append(l)
            x_val = np.concatenate(x_val, axis=0)
            y_val = np.array(y_val)
            print(x_val.shape, y_val.shape)
            pickle.dump((x_val, y_val), open('./dataset/%s_%s_val_%i.p' % (args.d, args.model, i), 'wb'), protocol=4)
            print('Random validation %i-th' % (i))
    print('Now you can load the validation dataset')
    exit()

def get_ats(model, dataset, layer_names):
    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    pred = model.predict(dataset)
    pred = np.argmax(pred, axis=1)

    if len(layer_names) == 1:            
            layer_outputs = [temp_model.predict(dataset)]
    else:            
        layer_outputs = temp_model.predict(dataset)    
    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        print("Layer: " + layer_name)
        if layer_output[0].ndim == 3:
            # For convolutional layers
            layer_matrix = np.array(
                p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
            )
        else:
            layer_matrix = np.array(layer_output)

        if ats is None:
            ats = layer_matrix
        else:
            ats = np.append(ats, layer_matrix, axis=1)
            layer_matrix = None
    return ats, pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
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
    """We have five different attacks:
        + Fast Gradient Sign Method (fgsm)
        + Basic Iterative Method (bim-a, bim-b, or bim)
        + Jacobian-based Saliency Map Attack (jsma)
        + Carlini&Wagner (c+w)
    """
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,        
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--layer",
        "-layer",
        help="Layer name",
        type=str,
    )
    parser.add_argument(
        "--random_train", "-random_train", help="random selected images for training (only for IMAGENET dataset)", action="store_true"
    )
    parser.add_argument("--random_train_size", "-random_train_size", type=int, default=1)
    parser.add_argument("--random_train_num", "-random_train_num", type=int, default=10)
    parser.add_argument(
        "--val", "-val", help="load validation dataset (only for IMAGENET dataset)", action="store_true"
    )
    parser.add_argument(
        "--val_size", "-val_size", help="Validation size used to chunk (only for IMAGENET dataset)", type=int, default=1000
    )
    parser.add_argument("--model", "-model", help="Model for IMAGENET dataset", type=str, default="densenet201")
    parser.add_argument("--random_train_ith", "-random_train_ith", type=str, default="0, 1")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.lsa ^ args.dsa ^ args.conf ^ args.true_label ^ args.pred_label ^ args.adv_lsa ^ args.adv_dsa ^ args.adv_conf, "Select either 'lsa' or 'dsa' or etc."
    print(args)

    # # model = efn.EfficientNetB7(weights='imagenet')
    # # for i in range(0, 10):
    # #     x, y = pickle.load(open('./dataset/%s_%s_random_train_%i.p' % (args.d, args.model, i), 'rb'))
    # #     y_pred = model.predict(x)
    # #     y_pred = np.argmax(y_pred, axis=1)
    # #     print(i, args.d, args.model)
    # #     write_file('./dataset/%s_%s_random_train_pred_%i.txt' % (args.d, args.model, i), y_pred)        
    # # exit()
    # # pred = load_file('./metrics/imagenet_efficientnetb7_pred_label.txt')
    # pred_0 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_0.txt')
    # pred_1 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_1.txt')
    # pred_2 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_2.txt')
    # pred_3 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_3.txt')
    # pred_4 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_4.txt')
    # pred_5 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_5.txt')
    # pred_6 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_6.txt')
    # pred_7 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_7.txt')
    # pred_8 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_8.txt')
    # pred_9 = load_file('./dataset/imagenet_efficientnetb7_random_train_pred_9.txt')
    # # pred_2 = pred[1974:(1974+1954)]
    # # pred_3 = pred[(1974+1954):(1974+1954+1958)]
    # # pred_4 = pred[(1974+1954+1958):(1974+1954+1958+1958)]
    # # pred_5 = pred[(1974+1954+1958+1958):]
    # # print(len(pred), len(pred_1), len(pred_2), len(pred_3), len(pred_4), len(pred_5))
    # # print(len(pred_1) + len(pred_2) + len(pred_3) + len(pred_4) + len(pred_5))
    # pred = set(pred_0 + pred_1 + pred_2)
    # print(len(pred))
    # pred = set(pred_3 + pred_4 + pred_5)
    # print(len(pred))
    # pred = set(pred_7 + pred_8 + pred_9)
    # print(len(pred))
    # exit()

    if args.d == 'imagenet':
        if args.model == 'efficientnetb0':
            model = efn.EfficientNetB0(weights='imagenet')        
        if args.model == 'efficientnetb7':
            model = efn.EfficientNetB7(weights='imagenet')    
        
        args.image_size = model.input_shape[1]
        args.num_classes = 1000
        
        if args.random_train == True:
            print('Loading training IMAGENET dataset -----------------------------')
            path_img_train = '../datasets/ilsvrc2012/images/train/'
            path_train_info = '../datasets/ilsvrc2012/images/train.txt' 
            load_imagenet_random_train(path_img=path_img_train, path_info=path_train_info, args=args)
              
        if args.val == True:
            print('Loading validation IMAGENET dataset -----------------------------')  
            path_img_val = '../datasets/ilsvrc2012/images/val/'
            path_val_info = '../datasets/ilsvrc2012/images/val.txt'
            load_imagenet_val(path_img=path_img_val, path_info=path_val_info, args=args)

        if args.conf == True:            
            for i in range(0, 50):
                x_test, y_test = pickle.load(open('./dataset/%s_%s_val_%i.p' % (args.d, args.model, i), 'rb'))
                y_pred = model.predict(x_test)
                print(i, x_test.shape, y_test.shape, y_pred.shape)
                y_pred = np.amax(y_pred, axis=1)
                write_file('./metrics/%s_%s_conf_val_%i.txt' % (args.d, args.model, i), y_pred)                
        
        if args.pred_label == True:
            for i in range(0, 50):
                x_test, y_test = pickle.load(open('./dataset/%s_%s_val_%i.p' % (args.d, args.model, i), 'rb'))
                y_pred = model.predict(x_test)
                print(i, x_test.shape, y_test.shape, y_pred.shape)
                y_pred = np.argmax(y_pred, axis=1)
                write_file('./metrics/%s_%s_pred_label_val_%i.txt' % (args.d, args.model, i), y_pred)

        if args.true_label == True:
            for i in range(0, 50):
                x_test, y_test = pickle.load(open('./dataset/%s_%s_val_%i.p' % (args.d, args.model, i), 'rb'))                
                write_file('./metrics/%s_%s_true_label_val_%i.txt' % (args.d, args.model, i), y_test)

        if args.lsa == True:            
            random_train_files = args.random_train_ith.split(',')                   
            train_ats, train_pred = list(), list()
            print('Loading training IMAGENET dataset -----------------------------')
            for i in random_train_files:
                x, y = pickle.load(open('./dataset/%s_%s_random_train_%i.p' % (args.d, args.model, int(i)), 'rb'))            
                print(i, x.shape, y.shape)
                ats, pred = get_ats(model=model, dataset=x, layer_names=[args.layer])                
                train_ats.append(ats)
                train_pred.append(pred)
            train_ats, train_pred = np.concatenate(train_ats, axis=0), np.concatenate(train_pred, axis=0)
            print(train_ats.shape, train_pred.shape)
            train = (train_ats, train_pred)

            class_matrix = {}
            if args.is_classification:
                for i, label in enumerate(train_pred):
                    if label not in class_matrix:
                        class_matrix[label] = []
                    class_matrix[label].append(i)

            kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, args)

            print('Loading validation IMAGENET dataset -----------------------------')
            for i in range(0, 50):
                x_test, y_test = pickle.load(open('./dataset/%s_%s_val_%i.p' % (args.d, args.model, i), 'rb'))
                print(i, x_test.shape, y_test.shape)
                test_ats, test_pred = get_ats(model=model, dataset=x_test, layer_names=[args.layer])
                test = test_ats, test_pred
                test_lsa = fetch_lsa_imagenet(model, kdes, removed_cols, test, args)
                write_file('./metrics/%s_%s_lsa_random_train_%s_val_%i.txt' % (args.d, args.model, args.random_train_ith, i), test_lsa)               
            