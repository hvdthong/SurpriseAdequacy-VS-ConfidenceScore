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

class VGG16SelfConfidClassic(nn.Module):
    def __init__(self):
        super(VGG16SelfConfidClassic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_dropout = nn.Dropout(0.4)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv5_dropout = nn.Dropout(0.4)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv6_dropout = nn.Dropout(0.4)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv8_dropout = nn.Dropout(0.4)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv9_dropout = nn.Dropout(0.4)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv11_dropout = nn.Dropout(0.4)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.conv12_dropout = nn.Dropout(0.4)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2)

        self.end_dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

        self.uncertainty1 = nn.Linear(512, 400)
        self.uncertainty2 = nn.Linear(400, 400)
        self.uncertainty3 = nn.Linear(400, 400)
        self.uncertainty4 = nn.Linear(400, 400)
        self.uncertainty5 = nn.Linear(400, 1)

    def forward(self, x):
        # import pdb; pdb.set_trace()

        out = F.relu(self.conv1(x))
        out = self.conv1_bn(out)        
        out = self.conv1_dropout(out)
        out = F.relu(self.conv2(out))
        
        out = self.conv2_bn(out)
        out = self.maxpool1(out)

        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)        
        out = self.conv3_dropout(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = self.maxpool2(out)

        out = F.relu(self.conv5(out))
        out = self.conv5_bn(out)        
        out = self.conv5_dropout(out)
        out = F.relu(self.conv6(out))
        out = self.conv6_bn(out)        
        out = self.conv6_dropout(out)
        
        out = F.relu(self.conv7(out))
        out = self.conv7_bn(out)
        out = self.maxpool3(out)

        out = F.relu(self.conv8(out))
        out = self.conv8_bn(out)        
        out = self.conv8_dropout(out)
        out = F.relu(self.conv9(out))
        out = self.conv9_bn(out)        
        out = self.conv9_dropout(out)
        out = F.relu(self.conv10(out))
        out = self.conv10_bn(out)
        out = self.maxpool4(out)

        out = F.relu(self.conv11(out))
        out = self.conv11_bn(out)        
        out = self.conv11_dropout(out)
        out = F.relu(self.conv12(out))
        out = self.conv12_bn(out)
        out = self.conv12_dropout(out)
        out = F.relu(self.conv13(out))
        out = self.conv13_bn(out)
        out = self.maxpool5(out)        
        out = self.end_dropout(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))        
        out = self.dropout_fc(out)

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

        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=args.download, transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=args.download, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.d == 'mnist' or args.d == 'cifar':
        if args.train_clf:
            if args.d == 'mnist':
                model = SmallConvNetMNISTSelfConfidClassic().to(device)
            if args.d == 'cifar':
                model = VGG16SelfConfidClassic().to(device)

            model = freeze_layers(model=model, freeze_uncertainty_layers=True)            

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()            

            if args.d == 'mnist':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
            if args.d == 'cifar':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)                
            general_acc = 0
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
                print('Epoch %i / %i -- Total loss: %f -- Accuracy on testing data: %.2f' % (epoch, args.epoch, total_loss, accuracy))                      

                if accuracy > general_acc:
                    general_acc = accuracy
                    path_save = './model_confidnet/%s/train_clf/' % (args.d)
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    torch.save(model.state_dict(), path_save + 'epoch_%i_acc-%.2f.pt' % (epoch, accuracy))
                    print('Model improvement -- Saving into: ', path_save + 'epoch_%i_acc-%.2f.pt' % (epoch, accuracy))                


        if args.train_uncertainty == True:
            if args.d == 'mnist':
                model = SmallConvNetMNISTSelfConfidClassic().to(device)
                model.load_state_dict(torch.load('./model_confidnet/%s/train_clf/epoch_420_acc-98.58.pt' % (args.d)))
            if args.d == 'cifar':
                model = VGG16SelfConfidClassic().to(device)
                model.load_state_dict(torch.load('./model_confidnet/%s/train_clf/epoch_183_acc-82.58.pt' % (args.d)))

            model = freeze_layers(model=model, freeze_uncertainty_layers=False)
            accuracy, roc_score = eval_uncertainty(model=model, test_loader=test_loader)            

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
                print('Epoch %i / %i -- Total loss: %f -- Accuracy on testing data: %.2f -- AUC on testing data: %.2f' 
                        % (epoch, args.epoch, total_loss, accuracy, roc_score))

                # if epoch % 50 == 0:
                #     path_save = './model_confidnet/%s/train_uncertainty/' % (args.d)
                #     if not os.path.exists(path_save):
                #         os.makedirs(path_save)
                #     torch.save(model.state_dict(), path_save + 'epoch_%i_acc-%.2f_auc-%.2f.pt' % (epoch, accuracy, roc_score))


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
        "--batch_size", "-batch_size", help="Batch size", type=int, default=64
    )
    parser.add_argument(
        "--epoch", "-epoch", help="Epoch", type=int, default=1000
    )
    parser.add_argument(
        "--download", "-download", help="Download dataset", action="store_true"
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar", 'imagenet'], "Dataset should be either 'mnist' or 'cifar' or 'imagenet'"
    print(args)

    if args.train_clf == False and args.train_uncertainty == False:
        print('You need to input the classifier for training')
        exit()

    if args.train_clf == True and args.train_uncertainty == True:
        print('Choose one classifier model for training')
        exit()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args)