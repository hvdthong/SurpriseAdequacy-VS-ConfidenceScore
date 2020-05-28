from os import listdir
from os.path import isfile, join
from train import loading_data
from parameters import read_args
from ultis import mini_batches
from evaluation import eval
from padding import padding_message, padding_commit_code, mapping_dict_msg, mapping_dict_code
from train import padding_data
import torch
import operator
from ultis import write_file, load_file
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde

def get_best_model(dataset, params):
    path_model = './snapshot/'
    if dataset == 'openstack':
        name_folder = '2020-05-04_09-43-53'
    elif dataset == 'qt':
        name_folder = '2020-05-04_11-16-52'
    else:
        print('You need to input correct dataset')
        exit()

    mypath = path_model + name_folder
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = list(sorted(onlyfiles))

    path_to_model = [path_model + name_folder + '/' + f for f in onlyfiles]

    train, test, dictionary = loading_data(project=dataset)
    ids_train, labels_train, msg_train, code_train = train

    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))    

    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')    

    pad_msg_test = padding_data(data=msg_test, dictionary=dict_msg, params=params, type='msg')
    pad_code_test = padding_data(data=code_test, dictionary=dict_code, params=params, type='code')

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels_train.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels_train.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
    # create and train the defect model
    model = DefectNet(args=params)

    if torch.cuda.is_available():
        model = model.cuda()
    
    dict_results = dict()

    for p in path_to_model:
        model.load_state_dict(torch.load(p))
        print(p)
        acc, prc, rc, f1, auc_ = eval(data=batches_test, model=model)
        dict_results[p] = auc_ 
    
    for i in dict_results:
        print (i, dict_results[i])

    print(max(dict_results.items(), key=operator.itemgetter(1))[0])


def get_batches_params(dataset, params):
    train, test, dictionary = loading_data(project=dataset)
    ids_train, labels_train, msg_train, code_train = train

    ids_test, labels_test, msg_test, code_test = test
    dict_msg, dict_code = dictionary
    print('Dictionary message: %i -- Dictionary code: %i' % (len(dict_msg), len(dict_code)))    

    pad_msg_train = padding_data(data=msg_train, dictionary=dict_msg, params=params, type='msg')
    pad_code_train = padding_data(data=code_train, dictionary=dict_code, params=params, type='code')    

    pad_msg_test = padding_data(data=msg_test, dictionary=dict_msg, params=params, type='msg')
    pad_code_test = padding_data(data=code_test, dictionary=dict_code, params=params, type='code')

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels_train.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels_train.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batches_test = mini_batches(X_msg=pad_msg_test, X_code=pad_code_test, Y=labels_test)
    batches_train = mini_batches(X_msg=pad_msg_train, X_code=pad_code_train, Y=labels_train)
    return batches_train, batches_test, params


def get_confidence_score(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        all_predict = list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                _, predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                _, predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
        return all_predict

def get_true_label(data):
    all_labels = list()
    for batch in data:
        pad_msg, pad_code, labels = batch        
        all_labels += list(labels)
    return all_labels

def get_pred_label(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        all_predict = list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                _, predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                _, predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
        all_predict = [1 if p >= 0.5 else 0 for p in all_predict]
        return all_predict

def get_lsa_information(data, model):
    with torch.no_grad():
        model.eval()  # since we use drop out
        features = list()
        all_predict = list()
        for batch in data:
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_code).cuda()
            else:
                pad_msg, pad_code = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long()
            if torch.cuda.is_available():
                ftr, predict = model.forward(pad_msg, pad_code)                
                ftr = ftr.cpu().detach().numpy()
                predict = predict.cpu().detach().numpy().tolist()
            else:
                ftr, predict = model.forward(pad_msg, pad_code)
                ftr = ftr.detach().numpy().tolist()   
                predict = predict.detach().numpy().tolist()         
            features.append(ftr)
            all_predict += predict
        features = np.concatenate(features)
        all_predict = [1 if p >= 0.5 else 0 for p in all_predict]
        all_predict = np.array(all_predict)        
        return features, all_predict

def get_lsa(data, model):
    train, test = data
    test_ats, test_pred = get_lsa_information(data=test, model=model)
    train_ats, train_pred = get_lsa_information(data=train, model=model)

    class_matrix = {}
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)

    from sklearn.decomposition import PCA  # using PCA to reduce the dimensions
    pca = PCA(n_components=10)
    pca.fit(train_ats)
    train_ats = pca.transform(train_ats)
    test_ats = pca.transform(test_ats)
    
    num_classes = len(set(list(train_pred)))    
    kdes = {}
    for label in tqdm(range(num_classes), desc="kde"):
        refined_ats = np.transpose(train_ats[class_matrix[label]])
        kdes[label] = gaussian_kde(refined_ats)

    test_lsa = []
    print("Fetching LSA")
    for i, at in enumerate(tqdm(test_ats)):
        label = test_pred[i]
        test_lsa.append(np.asscalar(-kdes[label].logpdf(np.transpose(at))))
    return test_lsa

def get_dsa(data, model):
    train, test = data
    test_ats, test_pred = get_lsa_information(data=test, model=model)
    train_ats, train_pred = get_lsa_information(data=train, model=model)

    from sklearn.decomposition import PCA  # using PCA to reduce the dimensions
    pca = PCA(n_components=10)
    pca.fit(train_ats)
    train_ats = pca.transform(train_ats)
    test_ats = pca.transform(test_ats)

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    from sa import find_closest_at
    dsa = []
    print("Fetching DSA")
    for i, at in enumerate(tqdm(test_ats)):
        label = test_pred[i]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa.append(a_dist / b_dist)
    return dsa

def get_trustscore(data, model):
    train, test = data
    test_ats, test_pred = get_lsa_information(data=test, model=model)
    train_ats, train_pred = get_lsa_information(data=train, model=model)

    train_label = np.array(get_true_label(data=train))

    # from sklearn.decomposition import PCA  # using PCA to reduce the dimensions
    # pca = PCA(n_components=10)
    # pca.fit(train_ats)
    # train_ats = pca.transform(train_ats)
    # test_ats = pca.transform(test_ats)
    print(train_ats.shape, test_ats.shape)

    from trust_score_example import trustscore
    trust_model = trustscore.TrustScore()
    trust_model.fit(train_ats, train_label)

    trust_score = trust_model.get_score(test_ats, test_pred).tolist()
    return trust_score


def get_AIreliable(dataset, params):
    if dataset == 'openstack':
        path_model = './snapshot/2020-05-04_09-43-53/epoch_50.pt'
    if dataset == 'qt':
        path_model = './snapshot/2020-05-04_11-16-52/epoch_15.pt'

    batches_train, batches_test, params = get_batches_params(dataset=dataset, params=params)
    print(params)

    if params.type_reliable != 'confidnet':
        from model_defect import DefectNet
        model = DefectNet(args=params)
    elif params.type_reliable == 'confidnet':
        from model_defect_confidnet import DefectNet
        model = DefectNet(args=params)
    else:
        print('You need to give correct name of reliable method')
        exit()
        
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(path_model))

    if params.type_reliable == 'conf':
        score = get_confidence_score(data=batches_test, model=model)
    if params.type_reliable == 'true_label':
        score = get_true_label(data=batches_test)
    if params.type_reliable == 'pred_label':
        score = get_pred_label(data=batches_test, model=model)
    if params.type_reliable == 'lsa':
        score = get_lsa(data=(batches_train, batches_test), model=model)
    if params.type_reliable == 'dsa':
        score = get_dsa(data=(batches_train, batches_test), model=model)
    if params.type_reliable == 'ts':
        score = get_trustscore(data=(batches_train, batches_test), model=model)
    write_file('../metrics/defect_%s_%s.txt' % (dataset, params.type_reliable), data=score)
                

if __name__ == '__main__':
    # dataset = 'openstack'
    # # dataset = 'qt'

    # input_option = read_args().parse_args()
    # input_help = read_args().print_help()

    # get_best_model(dataset=dataset, params=input_option)

    # dataset = 'openstack'
    dataset = 'qt'

    input_option = read_args().parse_args()
    input_help = read_args().print_help()
    # input_option.type_reliable = 'conf'
    # input_option.type_reliable = 'true_label'
    # input_option.type_reliable = 'pred_label'
    # input_option.type_reliable = 'lsa'
    # input_option.type_reliable = 'dsa'
    input_option.type_reliable = 'ts'
    get_AIreliable(dataset=dataset, params=input_option)
            
    