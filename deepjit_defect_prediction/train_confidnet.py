from train import loading_data, padding_data
from parameters import read_args
from split_train_test import info_label
from model_defect_confidnet import DefectNet
from ultis import mini_batches, mini_batches_update
import torch
import os 
import datetime
from ultis import write_file
import torch.nn as nn
from train import save
from evaluation import evaluation_metrics
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc


def evaluation_confidnet(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        all_predict, all_label = list(), list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                predict, uncertainty  = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict, uncertainty = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_pred=all_predict, y_true=all_label)
        # print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
        return acc, prc, rc, f1, auc_

def evaluation_uncertainty(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        all_predict, all_label = list(), list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                predict, uncertainty  = model.forward(pad_msg, pad_code)
                uncertainty = uncertainty.cpu().detach().numpy().tolist()
            else:
                predict, uncertainty = model.forward(pad_msg, pad_code)
                uncertainty = uncertainty.detach().numpy().tolist()
            all_predict += uncertainty
            all_label += labels.tolist()

        fpr, tpr, _ = roc_curve(all_label, all_predict)
        roc_auc_conf = auc(fpr, tpr)
        # print('AUC: %f' % (roc_auc_conf))
        return roc_auc_conf

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

def confid_mse_loss(input, target, args):
    # probs = F.softmax(input[0], dim=1)    
    probs = input[0]
    confidence = torch.sigmoid(input[1]).squeeze()

    # labels_hot = one_hot_embedding(target, args.class_num).to(args.device)
    labels_hot = 1
    # weights = torch.ones_like(target).type(torch.FloatTensor).to(args.device)
    # weights[(probs.argmax(dim=1) != target)] *= 1
    # weights = weights * 1

    # Apply optional weighting
    # loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2        
    loss = (confidence - (probs * labels_hot)) ** 2        
    return torch.mean(loss)

def train_confidnetnet_model(train, test, dictionary, params, options):
    #####################################################################################################
    # training model using 50% of positive and 50% of negative data in mini batch
    #####################################################################################################
    ids_train, labels_train, msg_train, code_train = train
    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))
    print('Training data')
    info_label(labels_train)

    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')
    print(pad_msg_train.shape, pad_code_train.shape)

    print('Testing data')
    info_label(labels_test)
    pad_msg_test = padding_data(data=msg_test, dictionary=dict_msg, params=params, type='msg')
    pad_code_test = padding_data(data=code_test, dictionary=dict_code, params=params, type='code')
    print(pad_msg_test.shape, pad_code_test.shape)

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels_train.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels_train.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if options == 'clf':
        # create and train the defect model
        model = DefectNet(args=params)
        if torch.cuda.is_available():
            model = model.cuda()

        model = freeze_layers(model=model, freeze_uncertainty_layers=True)

        # print('Training model with options', options)
        # for param in model.named_parameters():
        #     print(param[0], param[1].requires_grad)        

        optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
        steps = 0

        batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
        write_log = list()
        for epoch in range(1, params.num_epochs + 1):
            # building batches for training model
            batches_train = mini_batches_update(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
            for batch in batches_train:
                pad_msg, pad_code, labels = batch
                if torch.cuda.is_available():
                    pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                        pad_code).cuda(), torch.cuda.FloatTensor(labels)
                else:
                    pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                        labels).float()

                optimizer.zero_grad()
                predict, uncertainty = model.forward(pad_msg, pad_code)
                loss = nn.BCELoss()
                loss = loss(predict, labels)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % params.log_interval == 0:
                    print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

            print('Epoch: %i ---Training data' % (epoch))
            acc, prc, rc, f1, auc_ = evaluation_confidnet(data=batches_train, model=model)
            print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
            print('Epoch: %i ---Testing data' % (epoch))
            acc, prc, rc, f1, auc_ = evaluation_confidnet(data=batches_test, model=model)
            print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
            write_log.append('Epoch - testing: %i --- Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (epoch, acc, prc, rc, f1, auc_))
            if epoch % 5 == 0:
                save(model, params.save_dir, 'epoch', epoch)
        write_file(params.save_dir + '/log.txt', write_log)

    if options == 'confidnet':
        # create and train the defect model
        model = DefectNet(args=params)
        if torch.cuda.is_available():
            model = model.cuda()

        if params.project == 'openstack':
            model.load_state_dict(torch.load('./snapshot/2020-05-17_09-37-57/epoch_55.pt'), strict=True)
        if params.project == 'qt':
            model.load_state_dict(torch.load('./snapshot/2020-05-17_12-50-56/epoch_15.pt'), strict=True)

        model = freeze_layers(model=model, freeze_uncertainty_layers=False)
        
        print('Training model with options', options)
        for param in model.named_parameters():
            print(param[0], param[1].requires_grad)

        optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
        steps = 0

        batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
        write_log = list()
        for epoch in range(1, params.num_epochs + 1):
            # building batches for training model
            batches_train = mini_batches_update(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
            for batch in batches_train:
                pad_msg, pad_code, labels = batch
                if torch.cuda.is_available():
                    pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                        pad_code).cuda(), torch.cuda.FloatTensor(labels)
                else:
                    pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                        labels).float()

                optimizer.zero_grad()
                predict, uncertainty = model.forward(pad_msg, pad_code)
                loss = confid_mse_loss((predict, uncertainty), labels, args=params)
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % params.log_interval == 0:
                    print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

            print('Epoch: %i ---Training data' % (epoch))
            auc_ = evaluation_uncertainty(data=batches_train, model=model)
            print('AUC: %f' % (auc_))
            print('Epoch: %i ---Testing data' % (epoch))
            auc_ = evaluation_uncertainty(data=batches_test, model=model)
            print('AUC: %f' % (auc_))
            write_log.append('Epoch - testing: %i --- AUC: %f' % (epoch, auc_))

            if epoch % 5 == 0:
                save(model, params.save_dir, 'epoch', epoch)
        write_file(params.save_dir + '/log.txt', write_log)


if __name__ == '__main__':
    # project = 'openstack'
    project = 'qt'

    train, test, dictionary = loading_data(project=project)
    print(type(train), type(test), type(dictionary))

    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    input_option.project = project

    # options = 'clf'
    options = 'confidnet'
    train_confidnetnet_model(train=train, test=test, dictionary=dictionary, params=input_option, options=options)

