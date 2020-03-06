import argparse
from keras.datasets import mnist, cifar10
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from keras.utils import np_utils

def freeze_layers(model, freeze_uncertainty_layers=True):
    if freeze_uncertainty_layers == True:
        for param in model.named_parameters():
            if "uncertainty" in param[0]:
                param[1].requires_grad = False
    else:
        for param in model.named_parameters():
            if "uncertainty" not in param[0]:
                param[1].requires_grad = False
    return model

class SmallConvNetMNISTSelfConfidClassic(nn.Module):
    def __init__(self):
        super(SmallConvNetMNISTSelfConfidClassic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.uncertainty1 = nn.Linear(128, 400)
        self.uncertainty2 = nn.Linear(400, 400)
        self.uncertainty3 = nn.Linear(400, 400)
        self.uncertainty4 = nn.Linear(400, 400)
        self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)        
        out = self.dropout1(out)                        
        out = F.relu(self.fc1(out))        
        out = self.dropout2(out)
        uncertainty = F.relu(self.uncertainty1(out))
        uncertainty = F.relu(self.uncertainty2(uncertainty))
        uncertainty = F.relu(self.uncertainty3(uncertainty))
        uncertainty = F.relu(self.uncertainty4(uncertainty))
        uncertainty = self.uncertainty5(uncertainty)
        pred = self.fc2(out)
        return pred, uncertainty  

CLIP_MIN = -0.5
CLIP_MAX = 0.5

def train(args):
    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)

        x_train, y_train = torch.Tensor(x_train.reshape(-1, 1, 28, 28)), torch.Tensor(y_train)
        x_test, y_test = torch.Tensor(x_test.reshape(-1, 1, 28, 28)), torch.Tensor(y_test)

    elif args.d == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)

        x_train, y_train = torch.Tensor(x_train.reshape(-1, 3, 32, 32)), torch.Tensor(y_train)
        x_test, y_test = torch.Tensor(x_test.reshape(-1, 3, 32, 32)), torch.Tensor(y_test)
        
    train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)

    test = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    if args.d == 'mnist':
        if args.train_clf:
            model = SmallConvNetMNISTSelfConfidClassic().to(device)
            model = freeze_layers(model=model, freeze_uncertainty_layers=True)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()        
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

            print(args.epoch)
            exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)    
    parser.add_argument(
        "--train_clf", "-train_clf", help="Train the confidnet for classification model", action="store_true", default=True
    )
    parser.add_argument(
        "--train_uncertainty", "-train_uncertainty", help="Train the confidnet for uncertainty model", action="store_true", default=False
    )   
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--epoch", "-epoch", help="Epoch", type=int, default=450
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar' or 'imagenet'"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args)