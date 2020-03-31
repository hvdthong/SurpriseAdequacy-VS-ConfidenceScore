import torch
import argparse
import pickle
from torch.utils.data import TensorDataset, DataLoader
from train_model_imagenet_efficientnet_confidnet import load_train_and_test_loader
from barbar import Bar
import numpy as np
from utils import convert_predict_and_true_to_binary, write_file, load_file
from run import load_header_imagenet, load_img_imagenet
from torchvision import transforms
from PIL import Image


def confidnet_score(model, test_loader, args):
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
        uncertainties = torch.cat(uncertainties).cpu().detach().numpy().flatten().tolist()

        binary_predicted_true = convert_predict_and_true_to_binary(predicted=predict_label, true=groundtruth)           
        binary_predicted_true = [True if b == 1 else False for b in binary_predicted_true]        
        return uncertainties, binary_predicted_true


def convert_image(img, args):
    tfms = transforms.Compose([
                transforms.Resize(args.image_size), 
                transforms.CenterCrop(args.image_size), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    try:
        return tfms(img).unsqueeze(0)
    except: 
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    parser.add_argument(
        "--confidnet", "-confidnet", help="Confidnet score for Test Datasets (IMAGENET)", action="store_true",
    )
    parser.add_argument(
        "--adv", "-adv", help="Confidnet score for Adversarial Examples", action="store_true"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=16
    )
    parser.add_argument(
        "--model", "-model", help="Model for IMAGENET dataset", type=str, default='efficientnetb7'
    )
    parser.add_argument(
        "--val_start", "-val_start", help="Start validation index (only for IMAGENET dataset)", type=int, default=0
    )
    parser.add_argument(
        "--val_end", "-val_end", help="End validation index (only for IMAGENET dataset)", type=int, default=50
    )
    args = parser.parse_args()
    assert args.d in ['imagenet'], "Dataset should be 'imagenet'"
    print(args)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    if args.confidnet:
        
        if args.model == 'efficientnetb7':
            from efficientnet_pytorch_model.model import EfficientNet as efn
            model = efn.from_name('efficientnet-b7').to(args.device)
            model.load_state_dict(torch.load('./model_confidnet/%s_efficientnet-b7/train_uncertainty/epoch_1_acc-0.83_auc-0.82.pt' % (args.d)), strict=True)
            args.image_size = 600
        
        # if args.adv == False:
        #     path_img_val = '../datasets/ilsvrc2012/images/val/'
        #     path_val_info = '../datasets/ilsvrc2012/images/val.txt'
        #     img_name, img_label = load_header_imagenet(load_file(path_val_info))
                    
        #     model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        #     with torch.no_grad():
        #         confidnet_score, binary_predict_label = list(), list()
        #         for i, (n, l) in enumerate(zip(img_name, img_label)):                
        #             img = Image.open(path_img_val + n)
        #             img = convert_image(img, args)
        #             if img is not None:
        #                 img = torch.Tensor(img).to(args.device)
        #                 if len(img.shape) == 4 and img.shape[1] == 3:
        #                     pred, uncertainty = model(img)                
        #                     pred = pred.cpu().detach().numpy()
        #                     predict_label = np.argmax(pred, axis=1)[0]
        #                     if predict_label == int(l):
        #                         binary_predict_label.append('True')
        #                     else:
        #                         binary_predict_label.append('False')
                            
        #                     confidnet_score.append(uncertainty.cpu().detach().numpy().flatten().tolist()[0])
        #                     print(i, len(confidnet_score), len(binary_predict_label)) 
        #             else:
        #                 continue
        #     write_file(path_file='./metrics/{}_{}_confidnet_score.txt'.format(args.d, args.model), data=confidnet_score)
        #     write_file(path_file='./metrics/{}_{}_confidnet_accurate.txt'.format(args.d, args.model), data=binary_predict_label)

        if args.adv == False:
            confidnet, binary_predict_label = list(), list()
            for i in range(args.val_start, args.val_end):
                x_test, y_test = pickle.load(open('./dataset_imagenet/%s_%s_val_%i.p' % (args.d, args.model, i), 'rb'))                
                x_test = np.rollaxis(x_test, 3, 1)                  

                x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
                test = TensorDataset(x_test, y_test)
                test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
                print('Running at i-th', i)
                s, b = confidnet_score(model, test_loader, args)
                confidnet += s 
                binary_predict_label += b                                
            write_file(path_file='./metrics/{}_{}_confidnet_score.txt'.format(args.d, args.model), data=confidnet)
            write_file(path_file='./metrics/{}_{}_confidnet_accurate.txt'.format(args.d, args.model), data=binary_predict_label)