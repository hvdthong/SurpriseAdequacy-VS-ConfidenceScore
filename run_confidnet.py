import argparse
import torch
from keras.datasets import mnist, cifar10
from torch.utils.data import TensorDataset, DataLoader
from train_model_confidnet import SmallConvNetMNISTSelfConfidClassic, VGG16SelfConfidClassic
import numpy as np
from utils import convert_predict_and_true_to_binary, write_file
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from barbar import Bar


CLIP_MIN = -0.5
CLIP_MAX = 0.5

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.uncertainty1 = nn.Linear(512 * 7 * 7, 4096)
        self.uncertainty2 = nn.Linear(4096, 1000)
        self.uncertainty3 = nn.Linear(1000, 1000)        
        self.uncertainty4 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)        
        pred = self.classifier(x)

        uncertainty = F.relu(self.uncertainty1(x))
        uncertainty = F.relu(self.uncertainty2(uncertainty))
        uncertainty = F.relu(self.uncertainty3(uncertainty))        
        uncertainty = self.uncertainty4(uncertainty)
        return pred, uncertainty


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress):    
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm)).to(device)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)        
        model.load_state_dict(state_dict, strict=False)
    return model



def confidnet_score(model, test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct, total = 0, 0
        uncertainties, pred, groundtruth = list(), list(), list()
        for i, (x, y) in enumerate(Bar(test_loader)):
            x, y = x.to(device), y.to(device, dtype=torch.long)
            print(x.shape, y.shape)        
            outputs, uncertainty = model(x)
            pred.append(outputs)
            groundtruth.append(y)
            uncertainties.append(uncertainty)
            exit()
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
    parser.add_argument(
        "--model", "-model", help="Model for IMAGENET dataset", type=str, default='vgg16'
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
            model.load_state_dict(torch.load('./model_confidnet/%s/train_uncertainty/epoch_950_acc-0.99_auc-0.92.pt' % (args.d)))
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
        
        if args.d == 'imagenet':      
            # model = VGG().to(device)
            # model.load_state_dict(torch.load(models.vgg16(pretrained=True)))
            # print(type(model))

            # model = _vgg('vgg16', 'D', False, True, False)
            # print(type(model))

            # for param in model.named_parameters():
            #     print(param[0], param[1].requires_grad)
            
            model = models.vgg16(pretrained=True)

            for param in model.named_parameters():
                print(param[0], param[1].requires_grad)
