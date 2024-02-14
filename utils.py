import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, accuracy_score 
import os
import time
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC


class MSELoss(nn.Module):
    '''MSE loss that sums over output dimensions and allows weights.'''
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        if weights is not None:
            loss = torch.sum(weights * ((pred - target) ** 2), dim=-1)
        else:
            loss = torch.sum((pred - target) ** 2, dim=-1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
        
def get_optimizer(optimizer, params, lr):
    '''Get optimizer.'''
    if optimizer == 'SGD':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'Momentum':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'Adam':
        return optim.Adam(params, lr=lr)
    elif optimizer == 'Adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'RMSprop':
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError('unsupported optimizer: {}'.format(optimizer))
    
def get_activation(activation):
    '''Get activation function.'''
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation is None:
        return nn.Identity()
    else:
        raise ValueError('unsupported activation: {}'.format(activation))

def save_res_4kfolds_cv(y_pres, y_tests, score, file_name=None, verbose=False):
    """
    save experiment results for k-folds cross validation
    :param y_pres: predicted labels, k*Ntest
    :param y_tests: true labels, k*Ntest
    :param file_name:
    :return:
    """
    ca, oa, aa, kappa = [], [], [], []
    for y_p, y_t in zip(y_pres, y_tests):
        ca_, oa_, aa_, kappa_ = score(y_t, y_p)
        ca.append(np.asarray(ca_)), oa.append(np.asarray(oa_)), aa.append(np.asarray(aa_)),
        kappa.append(np.asarray(kappa_))
    ca = np.asarray(ca) * 100
    oa = np.asarray(oa) * 100
    aa = np.asarray(aa) * 100
    kappa = np.asarray(kappa) * 100
    ca_mean, ca_std = np.round(ca.mean(axis=0), 2), np.round(ca.std(axis=0), 2)
    oa_mean, oa_std = np.round(oa.mean(), 2), np.round(oa.std(), 2)
    aa_mean, aa_std = np.round(aa.mean(), 2), np.round(aa.std(), 2)
    kappa_mean, kappa_std = np.round(kappa.mean(), 2), np.round(kappa.std(), 2)
    if file_name is not None:
        file_name = 'scores.npz'
        np.savez(file_name, y_test=y_tests, y_pre=y_pres,
                    ca_mean=ca_mean, ca_std=ca_std,
                    oa_mean=oa_mean, oa_std=oa_std,
                    aa_mean=aa_mean, aa_std=aa_std,
                    kappa_mean=kappa_mean, kappa_std=kappa_std)
        print('the experiments have been saved in ', file_name)

    if verbose is True:
        print('---------------------------------------------')
        print('ca\t\t', '\taa\t\t', '\toa\t\t', '\tkappa\t\t')
        print(ca_mean, '+-', ca_std)
        print(aa_mean, '+-', aa_std)
        print(oa_mean, '+-', oa_std)
        print(kappa_mean, '+-', kappa_std)

    # return ca, oa, aa, kappa
    #import pdb; pdb.set_trace()
    return np.asarray([ca_mean, ca_std]), np.asarray([aa_mean, aa_std]), \
            np.asarray([oa_mean, oa_std]), np.asarray([kappa_mean, kappa_std])

def score(y_test, y_predicted):
        """
        calculate the accuracy and other criterion according to predicted results
        :param y_test:
        :param y_predicted:
        :return: ca, oa, aa, kappa
        """
        
        '''overall accuracy'''
        oa = accuracy_score(y_test, y_predicted)
        '''average accuracy for each classes'''
        n_classes = max([np.unique(y_test).__len__(), np.unique(y_predicted).__len__()])
        ca = []
        for c in np.unique(y_test):
            y_c = y_test[np.nonzero(y_test == c)]  # find indices of each classes
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        aa = ca.mean()

        '''kappa'''
        kappa = cohen_kappa_score(y_test, y_predicted)
        return ca, oa, aa, kappa

def classification_and_eval(selected_bands, x, y, times=10, test_size=0.9):
    total_num, bands = x.shape
    selected_img = x[:, selected_bands]
    estimator = [KNN(n_neighbors=3), SVC(C=1e5, kernel='rbf', gamma=1.)]
    estimator_pre, y_test_all = [[], []], []
    for i in range(times):  # repeat N times K-fold C
        train_ind, val_ind = train_test_split(range(total_num), test_size=0.9, random_state=i+1)
        train_dataset = selected_img[train_ind]
        train_label = y[train_ind]
        test_dataset = selected_img[val_ind]
        test_label = y[val_ind] 
        
        y_test_all.append(test_label)
        for c in range(len(estimator)):
            estimator[c].fit(train_dataset, train_label)
            y_pre = estimator[c].predict(test_dataset)
            estimator_pre[c].append(y_pre)
    score_dic = {'knn':{'ca':[], 'oa':[], 'aa':[], 'kappa':[]},
                 'svm': {'ca': [], 'oa': [], 'aa': [], 'kappa': []}
                 }
    key_ = ['knn', 'svm']
    for z in range(len(estimator)):
        ca, oa, aa, kappa = save_res_4kfolds_cv(estimator_pre[z], y_test_all, score, file_name=None, verbose=False)
        # score.append([oa, kappa, aa, ca])
        score_dic[key_[z]]['ca'] = ca
        score_dic[key_[z]]['oa'] = oa
        score_dic[key_[z]]['aa'] = aa
        score_dic[key_[z]]['kappa'] = kappa
    return score_dic

def create_logger(root_dir, des=''):
    root_output_dir = Path(root_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(exist_ok=True, parents=True)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, des)
    final_log_file = root_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger
