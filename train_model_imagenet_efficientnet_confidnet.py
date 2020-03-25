import argparse
from utils import load_file
import torch
import numpy as np
from run import load_img_imagenet
from efficientnet_pytorch import EfficientNet
from barbar import Bar
import torchvision.transforms as transforms
import torchvision.datasets as datasets    
from train_model_confidnet import freeze_layers
import torch.nn.functional as F
from train_model_confidnet import one_hot_embedding
from utils import convert_predict_and_true_to_binary
from sklearn.metrics import roc_curve, auc
import os 

def divide_chunks(data, n):
    for i in range(0, len(data), n):  
        yield data[i:i + n]

def generated_batch_image(data, args):
    data_path, data_label = list(), list()
    for d in data:        
        path_img, label_img = d.split()[0], int(d.split()[1])
        img_shape = load_img_imagenet(path_img, args)
        if len(img_shape.shape) == 4:
            data_path.append(img_shape)
            data_label.append(label_img)    
    if len(data_path) > 0:
        data_path = np.concatenate(data_path, axis=0)
        data_label = np.array(data_label)
        data_path = np.reshape(data_path, (-1, 3, args.image_size, args.image_size))
        return (data_path, data_label, True)
    else:
        return (data_path, data_label, False)

def load_train_and_test_loader(train_dir, test_dir, args):
    train_dir = train_dir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    test_dir = test_dir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([       
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),         
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def eval_no_uncertainty(model, test_loader, args):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct, total = 0, 0    
        for i, (x, y) in enumerate(Bar(test_loader)):
            x, y = x.to(args.device), y.to(args.device, dtype=torch.long)         
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        return 100 * correct / total

def eval(model, test_loader, args):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct, total = 0, 0    
        for i, (x, y) in enumerate(Bar(test_loader)):
            x, y = x.to(args.device), y.to(args.device, dtype=torch.long)
            outputs, uncertainty = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        return 100 * correct / total

def eval_uncertainty(model, test_loader, args):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct, total = 0, 0
        uncertainties, pred, groundtruth = list(), list(), list()
        for i, (x, y) in enumerate(Bar(test_loader)):   
            x, y = x.to(args.device), y.to(args.device, dtype=torch.long)            
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

def confid_mse_loss(input, target, args):
    probs = F.softmax(input[0], dim=1)    
    confidence = torch.sigmoid(input[1]).squeeze()

    labels_hot = one_hot_embedding(target, args.nb_classes).to(args.device)
    weights = torch.ones_like(target).type(torch.FloatTensor).to(args.device)
    weights[(probs.argmax(dim=1) != target)] *= 1

    # Apply optional weighting
    loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2        
    return torch.mean(loss)

def train(args):
    if args.d == 'imagenet' and args.train_uncertainty:
        train_dir = '../datasets/ilsvrc2012/images/train/'    
        test_dir = '../datasets/ilsvrc2012/images/val_loader/'    

        train_loader, test_loader = load_train_and_test_loader(train_dir, test_dir, args)
        
        if args.model == 'efficientnet-b7':
            from efficientnet_pytorch_model.model import EfficientNet as efn
            model = efn.from_name(args.model).to(args.device)
            state_dict = EfficientNet.from_pretrained(args.model).to(args.device).state_dict()
            model.load_state_dict(state_dict, strict=False)
        
        # model = freeze_layers(model=model, freeze_uncertainty_layers=False)  

        # for param in model.named_parameters():
        #     print(param[0], param[1].requires_grad)
        # exit()
        # accuracy = eval(model, test_loader, args)
        # print('Accuracy on testing data: %.2f' % (accuracy))

        model = freeze_layers(model=model, freeze_uncertainty_layers=False)
        # for param in model.named_parameters():
        #     print(param[0], param[1].requires_grad)
        # exit()

        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        for epoch in range(args.epoch):
            total_loss = 0
            for i, (x, y) in enumerate(Bar(train_loader)):
                x, y = x.to(args.device), y.to(args.device, dtype=torch.long)                        
                
                # Backward and optimize
                optimizer.zero_grad()

                # Forward pass
                outputs, uncertainty = model(x)                    
                loss = confid_mse_loss((outputs, uncertainty), y, args=args)
                loss.backward()
                total_loss += loss
                optimizer.step()
                # break
            print('Running evaluation for uncertainty')
            accuracy, roc_score = eval_uncertainty(model=model, test_loader=test_loader, args=args)
            print('Epoch %i / %i -- Total loss: %f -- Accuracy on testing data: %.2f -- AUC on testing data: %.2f' % (epoch, args.epoch, total_loss, accuracy, roc_score))            
                        
            path_save = './model_confidnet/%s_%s/train_uncertainty/' % (args.d, args.model)
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            torch.save(model.state_dict(), path_save + 'epoch_%i_acc-%.2f_auc-%.2f.pt' % (epoch, accuracy, roc_score))        
            # exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", required=True, type=str, default='imagenet')    
    parser.add_argument(
        "--train_clf", "-train_clf", help="Train the confidnet for classification model", action="store_true"
    )
    parser.add_argument(
        "--train_uncertainty", "-train_uncertainty", help="Train the confidnet for uncertainty model", action="store_true"
    )
    parser.add_argument(
        "--epoch", "-epoch", help="Epoch", type=int, default=100
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=32
    )
    parser.add_argument("--model", "-model", help="Model for IMAGENET dataset", type=str, default='efficientnet-b7')
    args = parser.parse_args()
    assert args.d in ['imagenet'], "Dataset name"    

    # Device configuration
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.image_size = EfficientNet.get_image_size(args.model)
    if args.d == 'imagenet':
        args.nb_classes = 1000
    print(args)
    # exit()

    if args.train_clf == False and args.train_uncertainty == False:
        print('You need to input the classifier for training')
        exit()
    
    if args.train_clf == True and args.train_uncertainty == True:
        print('Choose one classifier model for training')
        exit()
    train(args)