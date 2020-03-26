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
from keras.models import Model
from scipy.stats import gaussian_kde
from tqdm import tqdm

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
    random_train_file = list()
    for m in range(args.random_train_num_start, args.random_train_num_end):
        random.seed(m)
        # for i, n in enumerate(name_folders):
        #     random_name_file = sorted(random.sample(os.listdir(path_img + n), args.random_train_size))
        #     process_folder = [load_img_imagenet(path_img + n + '/' + r, args) for r in random_name_file]
        #     process_folder = [p for p in process_folder if len(p.shape) == 4]
        #     if len(process_folder) > 0:                
        #         random_train_file += random_name_file
        #     print(m, i)

        if os.path.exists('./dataset/%s_%s_random_train_%i.p' % (args.d, args.model, m)):                
            print('File exists in your directory')
        else:                
            x_random_train, y_random_train = list(), list()
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
            
            x_random_train = np.concatenate(x_random_train, axis=0)
            y_random_train = np.concatenate(y_random_train, axis=0)                
            pickle.dump((x_random_train, y_random_train), open('./dataset/%s_%s_random_train_%i.p' % (args.d, args.model, m), 'wb'), protocol=4)        
    write_file('./dataset/%s_%s_random_train_name_file.txt' % (args.d, args.model), random_train_file)
    print('Now you can load the training dataset')
    exit()

def load_imagenet_random_train_ver2(path_img, path_info, args):
    """Load IMAGENET dataset random training for calculating DSA or LSA
    
    Args:
        path_img (string): Folder of train dataset
        path_info (string): A file which includes the information of training data
        args: Keyboards arguments
    """

    imgnet_info = load_file(path_info)
    name_folders = [p.split('/')[0].strip() for p in imgnet_info]
    name_folders = list(sorted(set(name_folders)))    
    random_train_file = list()
    random.seed(0)
    for i, n in enumerate(name_folders):
        random_train_file, x_random_train, y_random_train = list(), list(), list()

        random_name_file = sorted(random.sample(os.listdir(path_img + n), args.random_train_size))
        process_folder = [load_img_imagenet(path_img + n + '/' + r, args) for r in random_name_file]
        zip_process = [(p, r) for p, r in zip(process_folder, random_name_file) if len(p.shape) == 4]
        process_folder = [p for (p, r) in zip_process]
        random_name_file = [r for (p, r) in zip_process]
        if len(process_folder) > 0:
            process_folder = np.concatenate(process_folder, axis=0)
            label_folder = get_label_imagenet(name_folder=n, imagnet_info=imgnet_info)                
            label_folder = np.array([label_folder for i in range(len(process_folder))])            

        print(len(random_name_file), process_folder.shape, label_folder.shape)
        print('Random training the folder %i-th which has name %s' % (i, n))
        pickle.dump((random_name_file, process_folder, label_folder), open('./dataset_imagenet/%s_%s_random_train_%i.p' % (args.d, args.model, i), 'wb'), protocol=4)
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
        "--ts", "-ts", help="True Score", action="store_true"
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
    parser.add_argument(
        "--adv", "-adv", help="Used Adversarial Examples", action="store_true"
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
        # default=1e-4,
        # default=0,        
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
    parser.add_argument(
        "--random_train_ats", "-random_train_ats", help="Activation Trace for training IMAGENET dataset", action="store_true"
    )
    parser.add_argument(
        "--random_train_label", "-random_train_label", help="Label of training IMAGENET dataset", action="store_true"
    )    
    parser.add_argument("--random_train_start", "-random_train_start", type=int, default=0)
    parser.add_argument("--random_train_end", "-random_train_end", type=int, default=150)
    parser.add_argument("--random_train_size", "-random_train_size", type=int, default=100)

    parser.add_argument("--random_train_num", "-random_train_num", type=int, default=10)
    parser.add_argument("--random_train_num_start", "-random_train_num_start", type=int, default=10)
    parser.add_argument("--random_train_num_end", "-random_train_num_end", type=int, default=100)
    parser.add_argument(
        "--lsa_kdes", "-lsa_kdes", help="Kernel Density Estimation of Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--val", "-val", help="load validation dataset (only for IMAGENET dataset)", action="store_true"
    )
    parser.add_argument(
        "--val_size", "-val_size", help="Validation size used to chunk (only for IMAGENET dataset)", type=int, default=1000
    )
    
    parser.add_argument(
        "--val_ats", "-val_ats", help="Activation Trace for validation IMAGENET dataset", action="store_true"
    )
    parser.add_argument(
        "--val_adv_ats", "-val_adv_ats", help="Activation Trace for adversarial examples of validation IMAGENET dataset", action="store_true"
    )
    parser.add_argument(
        "--val_start", "-val_start", help="Start validation index (only for IMAGENET dataset)", type=int, default=0
    )
    parser.add_argument(
        "--val_end", "-val_end", help="End validation index (only for IMAGENET dataset)", type=int, default=50
    )
    parser.add_argument("--model", "-model", help="Model for IMAGENET dataset", type=str, default="densenet201")
    parser.add_argument("--random_train_ith", "-random_train_ith", type=str, default="0, 1")
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim", 'jsma', 'c+w'], "Dataset should be either 'fgsm', 'bim', 'jsma', 'c+w'"
    assert args.val_adv_ats ^ args.ts ^ args.random_train ^ args.random_train_ats ^ args.random_train_label ^ args.val_ats ^ args.lsa ^ args.dsa ^ args.conf ^ args.true_label ^ args.pred_label ^ args.adv_lsa ^ args.adv_dsa ^ args.adv_conf, "Select either 'lsa' or 'dsa' or etc."
    print(args)

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
            # load_imagenet_random_train(path_img=path_img_train, path_info=path_train_info, args=args)
            load_imagenet_random_train_ver2(path_img=path_img_train, path_info=path_train_info, args=args)
              
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

        if args.random_train_ats:                
            print('Loading training IMAGENET dataset to create trace activation -----------------------------')
            train_ats, train_pred = list(), list()
            for i in range(args.random_train_start, args.random_train_end):
                if os.path.exists('./dataset_imagenet/%s_%s_random_train_ats_%s_%i.p' % (args.d, args.model, args.layer, i)):
                    ats, pred = pickle.load(open('./dataset_imagenet/%s_%s_random_train_ats_%s_%i.p' % (args.d, args.model, args.layer, i), 'rb'))
                    train_ats.append(ats)
                    train_pred.append(pred)
                    print(i, ats.shape, pred.shape)
                else:
                    _, x, y = pickle.load(open('./dataset_imagenet/%s_%s_random_train_%i.p' % (args.d, args.model, int(i)), 'rb'))                                
                    ats, pred = get_ats(model=model, dataset=x, layer_names=[args.layer])    
                    print(i, x.shape, y.shape, ats.shape, pred.shape)
                    train_ats.append(ats)
                    train_pred.append(pred)
                    pickle.dump((ats, pred), open('./dataset_imagenet/%s_%s_random_train_ats_%s_%i.p' % (args.d, args.model, args.layer, i), 'wb'), protocol=4)
            train_ats, train_pred = np.concatenate(train_ats, axis=0), np.concatenate(train_pred, axis=0)
            pickle.dump((train_ats, train_pred), open('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer), 'wb'), protocol=4)
            print(train_ats.shape, train_pred.shape)
            exit()

        if args.random_train_label:
            print('Loading training IMAGENET dataset label -----------------------------')
            random_label = list()
            if os.path.exists('./dataset_imagenet/%s_%s_random_train_label.txt' % (args.d, args.model)):
                random_label = load_file('./dataset_imagenet/%s_%s_random_train_label.txt' % (args.d, args.model))
                print(len(random_label))
            else:
                for i in range(args.random_train_start, args.random_train_end):
                    _, x, y = pickle.load(open('./dataset_imagenet/%s_%s_random_train_%i.p' % (args.d, args.model, int(i)), 'rb'))                    
                    print(i, len(y))
                    random_label += y.tolist()
            print(len(random_label))
            write_file('./dataset_imagenet/%s_%s_random_train_label.txt' % (args.d, args.model), random_label)
        
        if args.val_ats:                
            print('Loading validation IMAGENET dataset to create trace activation -----------------------------')
            test_ats, test_pred = list(), list()
            for i in range(args.val_start, args.val_end):
                if os.path.exists('./dataset_imagenet/%s_%s_val_ats_%s_%i.p' % (args.d, args.model, args.layer, i)):
                    ats, pred = pickle.load(open('./dataset_imagenet/%s_%s_val_ats_%s_%i.p' % (args.d, args.model, args.layer, i), 'rb'))
                    test_ats.append(ats)
                    test_pred.append(pred)
                    print(i, ats.shape, pred.shape)
                else:
                    x, y = pickle.load(open('./dataset_imagenet/%s_%s_val_%i.p' % (args.d, args.model, int(i)), 'rb'))                                
                    ats, pred = get_ats(model=model, dataset=x, layer_names=[args.layer])
                    print(i, x.shape, y.shape, ats.shape, pred.shape)
                    test_ats.append(ats)
                    test_pred.append(pred)
                    pickle.dump((ats, pred), open('./dataset_imagenet/%s_%s_val_ats_%s_%i.p' % (args.d, args.model, args.layer, i), 'wb'), protocol=4)
            test_ats, test_pred = np.concatenate(test_ats, axis=0), np.concatenate(test_pred, axis=0)
            pickle.dump((test_ats, test_pred), open('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer), 'wb'), protocol=4)
            print(test_ats.shape, test_pred.shape)
            exit()

        if args.val_adv_ats:                
            print('Loading validation IMAGENET dataset to create trace activation -----------------------------')
            test_ats, test_pred = list(), list()
            for i in range(args.val_start, args.val_end):
                if os.path.exists('./adv/%s_%s_%s_val_ats_%s_%i.p' % (args.d, args.model, args.attack, args.layer, int(i))):
                    ats, pred = pickle.load(open('./adv/%s_%s_%s_val_ats_%s_%i.p' % (args.d, args.model, args.attack, args.layer, i), 'rb'))
                    test_ats.append(ats)
                    test_pred.append(pred)
                    print(i, ats.shape, pred.shape)
                else:
                    x, y = np.load(open('./adv/%s_%s_%s_val_%i.npy' % (args.d, args.model, args.attack, int(i)), encoding='utf-8'))                                
                    ats, pred = get_ats(model=model, dataset=x, layer_names=[args.layer])
                    print(i, x.shape, y.shape, ats.shape, pred.shape)
                    test_ats.append(ats)
                    test_pred.append(pred)
                    pickle.dump((ats, pred), open('./adv/%s_%s_%s_val_ats_%s_%i.p' % (args.d, args.model, args.attack, args.layer, i), 'wb'), protocol=4)
            test_ats, test_pred = np.concatenate(test_ats, axis=0), np.concatenate(test_pred, axis=0)
            pickle.dump((test_ats, test_pred), open('./adv/%s_%s_%s_val_ats_%s.p' % (args.d, args.model, args.attack, args.layer), 'wb'), protocol=4)
            print(test_ats.shape, test_pred.shape)
            exit()

        if args.lsa == True: 
            if args.random_train_ats == False and args.val_ats == False and args.val_adv_ats == False:
                if os.path.exists('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer)):
                    print('File exists in your directory')
                    (train_ats, train_pred) = pickle.load(open('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer), 'rb'))                  
                    print(train_ats.shape, train_pred.shape)
                else:
                    print('Please load the activation trace of training IMAGENET dataset')
                    exit()

                if args.adv == False:
                    if os.path.exists('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer)):
                        print('File exists in your directory')
                        (test_ats, test_pred) = pickle.load(open('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer), 'rb'))                  
                        print(test_ats.shape, test_pred.shape)
                    else:
                        print('Please load the activation trace of validation IMAGENET dataset')
                        exit()

                if args.adv == True:
                    if os.path.exists('./adv/%s_%s_%s_val_ats_%s.p' % (args.d, args.model, args.attack, args.layer)):
                        print('File exists in your directory')
                        (test_ats, test_pred) = pickle.load(open('./adv/%s_%s_%s_val_ats_%s.p' % (args.d, args.model, args.attack, args.layer), 'rb'))                  
                        print(test_ats.shape, test_pred.shape)
                    else:
                        print('Please load the activation trace of validation IMAGENET dataset')
                        exit()
                
                from sklearn.decomposition import PCA  # using PCA to reduce the dimensions
                pca = PCA(n_components=10)
                pca.fit(train_ats)
                train_ats = pca.transform(train_ats)
                test_ats = pca.transform(test_ats)                

                class_matrix = {}
                for i, label in enumerate(train_pred):
                    if label not in class_matrix:
                        class_matrix[label] = []
                    class_matrix[label].append(i)

                kdes = {}
                for label in tqdm(range(args.num_classes), desc="kde"):
                    refined_ats = np.transpose(train_ats[class_matrix[label]])
                    kdes[label] = gaussian_kde(refined_ats)
                
                test_lsa = []
                print("Fetching LSA")
                for i, at in enumerate(tqdm(test_ats)):
                    label = test_pred[i]
                    test_lsa.append(np.asscalar(-kdes[label].logpdf(np.transpose(at))))
                write_file('./metrics/%s_%s_lsa_%s.txt' % (args.d, args.model, args.layer), test_lsa)
                exit()

        if args.dsa == True:
            if args.random_train_ats == False and args.val_ats == False and args.val_adv_ats == False:
                if os.path.exists('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer)):
                    print('File exists in your directory')
                    (train_ats, train_pred) = pickle.load(open('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer), 'rb'))                  
                    print(train_ats.shape, train_pred.shape)
                else:
                    print('Please load the activation trace of training IMAGENET dataset')
                    exit()

                if args.adv == False:
                    if os.path.exists('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer)):
                        print('File exists in your directory')
                        (test_ats, test_pred) = pickle.load(open('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer), 'rb'))                  
                        print(test_ats.shape, test_pred.shape)
                    else:
                        print('Please load the activation trace of validation IMAGENET dataset')
                        exit()

                if args.adv == True:
                    if os.path.exists('./adv/%s_%s_%s_val_ats_%s.p' % (args.d, args.model, args.attack, args.layer)):
                        print('File exists in your directory')
                        (test_ats, test_pred) = pickle.load(open('./adv/%s_%s_%s_val_ats_%s.p' % (args.d, args.model, args.attack, args.layer), 'rb'))                  
                        print(test_ats.shape, test_pred.shape)
                    else:
                        print('Please load the activation trace of validation IMAGENET dataset')
                        exit()
                
                from sklearn.decomposition import PCA  # using PCA to reduce the dimensions
                pca = PCA(n_components=100)
                pca.fit(train_ats)
                train_ats = pca.transform(train_ats)
                test_ats = pca.transform(test_ats)
                print(train_ats.shape, test_ats.shape)

                class_matrix = {}
                all_idx = []
                for i, label in enumerate(train_pred):
                    if label not in class_matrix:
                        class_matrix[label] = []
                    class_matrix[label].append(i)
                    all_idx.append(i)

                from sa import find_closest_at
                dsa = []
                print("Fetching DSA")
                for i, at in enumerate(tqdm(test_ats)):
                    label = test_pred[i]
                    a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
                    b_dist, _ = find_closest_at(
                        a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
                    )
                    dsa.append(a_dist / b_dist)
                write_file('./metrics/%s_%s_dsa_%s.txt' % (args.d, args.model, args.layer), dsa)
                exit()

        if args.ts == True:
            if args.random_train_ats == False and args.val_ats == False and args.random_train_label == False:
                if os.path.exists('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer)):
                    print('File exists in your directory')
                    (train_ats, train_pred) = pickle.load(open('./dataset_imagenet/%s_%s_random_train_ats_%s.p' % (args.d, args.model, args.layer), 'rb'))                  
                    print(train_ats.shape, train_pred.shape)
                else:
                    print('Please load the activation trace of training IMAGENET dataset')
                    exit()

                if os.path.exists('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer)):
                    print('File exists in your directory')
                    (test_ats, test_pred) = pickle.load(open('./dataset_imagenet/%s_%s_val_ats_%s.p' % (args.d, args.model, args.layer), 'rb'))                  
                    print(test_ats.shape, test_pred.shape)
                else:
                    print('Please load the activation trace of validation IMAGENET dataset')
                    exit()

                if os.path.exists('./dataset_imagenet/%s_%s_random_train_label.txt' % (args.d, args.model)):
                    print('File exists in your directory')
                    train_label = load_file('./dataset_imagenet/%s_%s_random_train_label.txt' % (args.d, args.model))
                    train_label = np.array([int(l) for l in train_label])
                    print(train_label.shape)
                else:
                    print('Please load the activation trace of validation IMAGENET dataset')
                    exit()

                from sklearn.decomposition import PCA  # using PCA to reduce the dimensions
                pca = PCA(n_components=100)
                pca.fit(train_ats)
                train_ats = pca.transform(train_ats)
                test_ats = pca.transform(test_ats)
                print(train_ats.shape, test_ats.shape)

                from trust_score_example import trustscore
                trust_model = trustscore.TrustScore()
                trust_model.fit(train_ats, train_label)

                trust_score = trust_model.get_score(test_ats, test_pred).tolist()        
                write_file('./metrics/%s_%s_ts_%s.txt' % (args.d, args.model, args.layer), trust_score)




            