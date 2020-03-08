import argparse
import torch
from keras.datasets import mnist, cifar10
from torch.utils.data import TensorDataset, DataLoader
from train_model_confidnet import SmallConvNetMNISTSelfConfidClassic, VGG16SelfConfidClassic
import numpy as np
from utils import convert_predict_and_true_to_binary, write_file
import torchvision.transforms as transforms
import torchvision

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def confidnet_score(model, test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct, total = 0, 0
        uncertainties, pred, groundtruth = list(), list(), list()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device, dtype=torch.long)            
            outputs, uncertainty = model(x)
            pred.append(outputs)
            groundtruth.append(y)
            uncertainties.append(uncertainty)
        pred = torch.cat(pred).cpu().detach().numpy()
        predict_label = np.argmax(pred, axis=1)
        groundtruth = torch.cat(groundtruth).cpu().detach().numpy()
        uncertainties = torch.cat(uncertainties).cpu().detach().numpy().flatten()

        binary_predicted_true = convert_predict_and_true_to_binary(predicted=predict_label, true=groundtruth)           
        binary_predicted_true = [True if b == 1 else False for b in binary_predicted_true]        
        return uncertainties, binary_predicted_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    parser.add_argument(
        "--confidnet", "-confidnet", help="Confidnet score for Test Datasets (i.e., MNIST or CIFAR)", action="store_true",
    )
    parser.add_argument(
        "--adv_confidnet", "-adv_confidnet", help="Confidnet score for Adversarial Examples", action="store_true"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=64
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar' or 'imagenet'"
    print(args)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.confidnet:
        if args.d == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")
            x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
            x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

            x_train, y_train = torch.Tensor(x_train.reshape(-1, 1, 28, 28)), torch.Tensor(y_train)
            x_test, y_test = torch.Tensor(x_test.reshape(-1, 1, 28, 28)), torch.Tensor(y_test)

            train = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
            
            test = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

        elif args.d == 'cifar':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform_train)
            train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
            test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)            
        
        if args.d == 'mnist':
            model = SmallConvNetMNISTSelfConfidClassic().to(device)
            model.load_state_dict(torch.load('./model_confidnet/%s/train_uncertainty/epoch_950_acc-0.99_auc-0.92.pt'% (args.d)))
            if args.confidnet == True:
                score, accurate = confidnet_score(model=model, test_loader=test_loader)
                write_file(path_file='./metrics/{}_confidnet_score.txt'.format(args.d), data=score)
                write_file(path_file='./metrics/{}_confidnet_accurate.txt'.format(args.d), data=accurate)

        if args.d == 'cifar':
            model = VGG16SelfConfidClassic().to(device)
            model.load_state_dict(torch.load('./model_confidnet/%s/train_uncertainty/epoch_950_acc-0.84_auc-0.87.pt' % (args.d)))

            if args.confidnet == True:
                score, accurate = confidnet_score(model=model, test_loader=test_loader)
                write_file(path_file='./metrics/{}_confidnet_score.txt'.format(args.d), data=score)
                write_file(path_file='./metrics/{}_confidnet_accurate.txt'.format(args.d), data=accurate)