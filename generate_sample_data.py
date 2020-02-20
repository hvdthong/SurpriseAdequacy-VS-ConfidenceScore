from keras.datasets import mnist, cifar10
import argparse
from sklearn.model_selection import StratifiedKFold
import pickle
import os


def save_random_sample_data(args, save_path):
    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

    elif args.d == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if not os.path.exists(save_path):
        os.makedirs(save_path)    

    for t in range(args.r):
        skf = StratifiedKFold(n_splits=args.skf, random_state=t)        
        for random_index, _ in skf.split(x_train, y_train):            
            rand_x_train, rand_y_train = x_train[random_index], y_train[random_index]            
            break
        
        data = (rand_x_train, rand_y_train), (x_test, y_test)
        with open(save_path + '/' + '%i.pickle' % (t), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)       


def load_random_sample_data(save_path, ntime):
    with open(save_path + '/' + '%i.pickle' % (ntime), 'rb') as handle:
        data = pickle.load(handle)        
        return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--skf", "-skf", help="How many stratified k fold", type=int, default=4)
    parser.add_argument("--r", "-r", help="How many times we want to sample dataset", type=int, default=100)
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    print(args)

    save_path = '../2020_FSE_Empirical/%s' % args.d 
    save_random_sample_data(args=args, save_path=save_path)
    load_random_sample_data(save_path, 0)
    
    
