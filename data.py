# This is from https://github.com/GiorgioMorales/HSI-BandSelection/
import os
import numpy as np
from operator import truediv
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#from keras.utils import np_utils
from sklearn.preprocessing import minmax_scale

def loadata(name, compressed):
    data_path = os.path.join(os.getcwd(), 'datasets')
    comp = ''
    if compressed:
        comp = '_compressed'

    if name == 'IP':
        if compressed:
            dat = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected_compressed.mat'))['indian_pines']
        else:
            dat = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        return dat, label
    elif name == 'SA':
        dat = sio.loadmat(os.path.join(data_path, 'Salinas_corrected' + comp + '.mat'))['salinas_corrected']
        label = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        return dat, label
    elif name == 'PU':
        dat = sio.loadmat(os.path.join(data_path, 'PaviaU' + comp + '.mat'))['paviaU']
        label = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        return dat, label
    elif name == 'KSC':
        dat = sio.loadmat(os.path.join(data_path, 'KSC' + comp + '.mat'))['KSC']
        label = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        return dat, label
    elif name == 'BSW':
        dat = sio.loadmat(os.path.join(data_path, 'Botswana' + comp + '.mat'))['Botswana']
        label = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        return dat, label


def padWithZeros(Xc, margin=2):
    newX = np.zeros((Xc.shape[0] + 2 * margin, Xc.shape[1] + 2 * margin, Xc.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:Xc.shape[0] + x_offset, y_offset:Xc.shape[1] + y_offset, :] = Xc
    return newX

def createImageCubes(Xc, yc, window=5, removeZeroLabels=True):
    margin = int((window - 1) / 2)
    zeroPaddedX = padWithZeros(Xc, margin=margin)
    # split patches
    patchesData = np.zeros((Xc.shape[0] * Xc.shape[1], window, window, Xc.shape[2]))
    patchesLabels = np.zeros((Xc.shape[0] * Xc.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = yc[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def applyPCA(Xc, numComponents=75):
    newX = np.reshape(Xc, (-1, Xc.shape[2]))
    pcaC = PCA(n_components=numComponents, whiten=True)
    newX = pcaC.fit_transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[1], numComponents))
    return newX, pcaC


def AA_andEachClassAccuracy(confusion_m):
    list_diag = np.diag(confusion_m)
    list_raw_sum = np.sum(confusion_m, axis=1)
    each_ac = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_ac)
    return each_ac, average_acc


def splitraintestset(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def get_input_data(data_type, x, y):
    data_dict = {
        "pixel": get_content_pixels(x, y),
        "original": get_all_pixels(x, y),
        "cubes": createImageCubes(x, y) 
    }
    return data_dict[data_type]
    
def get_content_pixels(x, y):
    y_vec = y.reshape(-1)
    x_mat = x.reshape(x.shape[0]*x.shape[1], x.shape[2])

    ind = y_vec.nonzero()
    y_con = y_vec[ind]
    x_con = x_mat[ind]
    return x_con, y_con

def get_all_pixels(x, y):
    y_vec = y.reshape(-1)
    x_mat = x.reshape(x.shape[0]*x.shape[1], x.shape[2]) 
    return x_mat, y_vec

def get_background_pixels(x, y):
    y_vec = y.reshape(-1)
    x_mat = x.reshape(x.shape[0]*x.shape[1], x.shape[2])

    ind = y_vec.zero()
    y_background = y_vec[ind]
    x_background = x_mat[ind]
    return x_background, y_background

if __name__ == "__main__":
    data, label = loadata('IP', False)
    n_row, n_column, n_band = data.shape
    norm_data = minmax_scale(data.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    pixel_data, pixel_label = get_content_pixels(norm_data, label) 
    patch_data, patch_label = createImageCubes(norm_data, label) 
    #onehot_label = np_utils.to_categorical(patch_label)
    print(patch_data.shape)
    print(patch_label.shape)
    print(pixel_data.shape)
