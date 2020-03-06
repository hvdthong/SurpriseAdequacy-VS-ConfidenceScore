import argparse
from keras.datasets import mnist, cifar10
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from keras.utils import np_utils
from barbar import Bar
import os 
from utils import convert_predict_and_true_to_binary
import numpy as np 
from sklearn.metrics import roc_curve, auc


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


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def confid_mse_loss(input, target, nb_classes):
    probs = F.softmax(input[0], dim=1)    
    confidence = torch.sigmoid(input[1]).squeeze()

    labels_hot = one_hot_embedding(target, nb_classes).to(device)
    weights = torch.ones_like(target).type(torch.FloatTensor).to(device)
    weights[(probs.argmax(dim=1) != target)] *= 1

    # Apply optional weighting
    loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2        
    return torch.mean(loss)


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
        out = out.view(out.size(0), -1)                 
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

def eval(model, test_loader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct, total = 0, 0    
        for x, y in test_loader:
            x, y = x.to(device), y.to(device, dtype=torch.long)            
            outputs, uncertainty = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        return 100 * correct / total

def eval_uncertainty(model, test_loader):
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
        accuracy = sum(binary_predicted_true) / len(pred)

        fpr, tpr, _ = roc_curve(binary_predicted_true, uncertainties)
        roc_auc_conf = auc(fpr, tpr)        

        return accuracy, roc_auc_conf


def train(args):
    if args.d == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

        x_train, y_train = torch.Tensor(x_train.reshape(-1, 1, 28, 28)), torch.Tensor(y_train)
        x_test, y_test = torch.Tensor(x_test.reshape(-1, 1, 28, 28)), torch.Tensor(y_test)

    elif args.d == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

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

            for epoch in range(args.epoch):
                total_loss = 0
                for i, (x, y) in enumerate(Bar(train_loader)):
                    x, y = x.to(device), y.to(device, dtype=torch.long)
                    
                    # Backward and optimize
                    optimizer.zero_grad()

                    # Forward pass
                    outputs, uncertainty = model(x)                    
                    loss = criterion(outputs, y)
                    loss.backward()
                    total_loss += loss
                    optimizer.step()
                accuracy = eval(model=model, test_loader=test_loader)
                print('Epoch %i -- Total loss: %f -- Accuracy on testing data: %f' % (epoch, total_loss, round(accuracy, 3)))                

                if epoch % 50 == 0:
                    path_save = './model_confidnet/%s/train_clf/' % (args.d)
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    torch.save(model.state_dict(), path_save + 'epoch_%i_acc-%.2f.pt' % (epoch, accuracy))

        if args.train_uncertainty == True:
            model = SmallConvNetMNISTSelfConfidClassic().to(device)
            model.load_state_dict(torch.load('./model_confidnet/%s/train_clf/epoch_420_acc-98.58.pt'% (args.d)))
            model = freeze_layers(model=model, freeze_uncertainty_layers=False)            

            # Loss and optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
            for epoch in range(args.epoch):
                total_loss = 0
                for i, (x, y) in enumerate(Bar(train_loader)):
                    x, y = x.to(device), y.to(device, dtype=torch.long)
                    
                    # Backward and optimize
                    optimizer.zero_grad()

                    # Forward pass
                    outputs, uncertainty = model(x)                    
                    loss = confid_mse_loss((outputs, uncertainty), y, nb_classes=10)
                    loss.backward()
                    total_loss += loss
                    optimizer.step()

                accuracy, roc_score = eval_uncertainty(model=model, test_loader=test_loader)
                print('Epoch %i -- Total loss: %f -- Accuracy on testing data: %.2f -- AUC on testing data: %.2f' 
                        % (epoch, total_loss, accuracy, roc_score))

                if epoch % 50 == 0:
                    path_save = './model_confidnet/%s/train_uncertainty/' % (args.d)
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    torch.save(model.state_dict(), path_save + 'epoch_%i_acc-%.2f_auc-%.2f.pt' % (epoch, accuracy, roc_score))

    if args.d == 'cifar':
        print('hello')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)    
    parser.add_argument(
        "--train_clf", "-train_clf", help="Train the confidnet for classification model", action="store_true"
    )
    parser.add_argument(
        "--train_uncertainty", "-train_uncertainty", help="Train the confidnet for uncertainty model", action="store_true"
    )   
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--epoch", "-epoch", help="Epoch", type=int, default=1000
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar' or 'imagenet'"
    print(args)

    if args.train_clf == False and args.train_uncertainty == False:
        print('You need to input the classifier for training')
        exit()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args)