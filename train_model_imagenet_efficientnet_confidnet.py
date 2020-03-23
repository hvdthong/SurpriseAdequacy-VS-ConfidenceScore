import argparse
from utils import load_file
import torch
import numpy as np
from run import load_img_imagenet
from efficientnet_pytorch import EfficientNet
from barbar import Bar
import torchvision.transforms as transforms
import torchvision.datasets as datasets    

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

def train(args):
    if args.d == 'imagenet' and args.train_uncertainty:
        train_dir = '../datasets/ilsvrc2012/images/train/'    
        test_dir = '../datasets/ilsvrc2012/images/val_loader/'    

        train_loader, test_loader = load_train_and_test_loader(train_dir, test_dir, args)
        if args.model == 'efficientnet-b7':
            model = EfficientNet.from_pretrained(args.model).to(args.device)
                
        accuracy = eval_no_uncertainty(model, test_loader, args)
        print('Accuracy on testing data: %.2f' % (accuracy))    

            # with torch.no_grad():    
            #     for i, (x, y) in enumerate(Bar(test_loader)):
            #         x, y = x.to(device), y.to(device, dtype=torch.long)
            #         # print(x.shape, y.shape) 
            #         y_pred = model(x)
            #         print(y_pred.shape)
            #         exit()

        # train = load_file('./dataset_imagenet/%s_training.txt' % (args.d))
        # train = divide_chunks(train, args.batch_size)

        # val = load_file('./dataset_imagenet/%s_val.txt' % (args.d))
        # val = divide_chunks(val, args.batch_size)

        # if args.model == 'efficientnet-b7':
        #     model = EfficientNet.from_pretrained(args.model)
        #     model.eval()
        # with torch.no_grad():    
        #     for i, v in enumerate(val):                
        #         x, y, flag = generated_batch_image(v, args)
        #         print(x.shape)              
        #         if flag == True:
        #             y_pred = model(x)
        #             print(y_pred.shape)
        #         exit()


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
        "--batch_size", "-batch_size", help="Batch size", type=int, default=16
    )
    parser.add_argument("--model", "-model", help="Model for IMAGENET dataset", type=str, default='efficientnet-b7')
    args = parser.parse_args()
    assert args.d in ['imagenet'], "Dataset name"    

    # Device configuration
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.image_size = EfficientNet.get_image_size(args.model)
    print(args)

    if args.train_clf == False and args.train_uncertainty == False:
        print('You need to input the classifier for training')
        exit()
    
    if args.train_clf == True and args.train_uncertainty == True:
        print('Choose one classifier model for training')
        exit()
    train(args)