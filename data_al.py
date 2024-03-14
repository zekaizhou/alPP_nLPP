import os
from sklearn import svm
from sklearn import preprocessing
from shutil import copy, rmtree
import random
from skimage import io, util
import numpy as np
from aLPP import alpp, alpp_it
from nlpp_n import nlpp_n
from PCA import pca
# from LDA import lda
# from PCA_L1 import PCA_L1
# from LDA_L1 import LDA_L1
# from LRE import LRE
from LPP import lpp, cal_pairwise_dist
# from LR_LPP import LR_LPP
from KNN import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from nLPP import nlpp
from PIL import ImageFilter
import cv2
from sklearn.linear_model import Perceptron


def Buid_Train_Test(path1, path2, train_num):
    train_path = os.path.join(path2, "train")
    test_path = os.path.join(path2, "test")
    if os.path.exists(train_path) and os.path.exists(test_path):
        rmtree(train_path)
        rmtree(test_path)
        os.mkdir(train_path)
        os.mkdir(test_path)
    else:
        os.mkdir(train_path)
        os.mkdir(test_path)
    dir = os.listdir(path1)
    for i in dir:
        img_path = os.path.join(path1, i)
        img_dir = os.listdir(img_path)
        # 随机采样验证集的索引
        eval_index = random.sample(img_dir, train_num)
        for image in img_dir:
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(img_path, image)
                new_path = os.path.join(train_path, image)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(img_path, image)
                new_path = os.path.join(test_path, image)
                copy(image_path, new_path)
    return train_path, test_path


def Data_create(data_path):
    str = data_path + '/*.jpg'
    coll = io.ImageCollection(str)
    Data = np.zeros(shape=(1024, 1))
    for i in range(len(coll)):
        dst = np.reshape(coll[i], (1024, 1))
        if i == 0:
            Data = dst
        else:
            Data = np.append(Data, dst, axis=1)
    return Data


def Data_create_train(n, data_path):
    str = data_path + '/*.jpg'
    coll = io.ImageCollection(str)
    Data = np.zeros(shape=(1024, 1))
    for i in range(len(coll)):
        if (i % 20) < n:
            img = util.random_noise(coll[i], mode='s&p', amount=0.3)

            # cv2.imshow('sss', img)
            # cv2.waitKey(100)
            dst = np.reshape(img, (1024, 1))
            if i == 0:
                img3 = coll[i]
                img2 = img
                Data = dst
                img3 = np.array(img3)
                img2 = np.array(img2)
            else:
                img3 = np.append(img3, coll[i], axis=1)
                img2 = np.append(img2, img, axis=1)
                Data = np.append(Data, dst, axis=1)
        else:
            dst = np.reshape(coll[i], (1024, 1))
            Data = np.append(Data, dst, axis=1)
    # cv2.imshow('sss', img2)
    # cv2.imshow('ttt', img3)
    # cv2.waitKey(0)
    return Data


def data_Analyze(n, lam, path1, path2, ndim, k, train_num, test_num, beta,
                 MaxN):
    min_max_scaler = preprocessing.MinMaxScaler()
    path_Train, path_Test = Buid_Train_Test(path1, path2, train_num)
    Data_train = Data_create_train(n, path_Train)
    # Data_train = min_max_scaler.fit_transform(Data_train)
    Label_train = np.zeros(Data_train.shape[1])
    # Data_train = Data_train.astype(np.float32)
    for i in range(Label_train.size):
        Label_train[i] = int(i / train_num)

    # clusters = np.unique(Label_train)
    # for i in clusters:
    #     datai = Data_train.T[Label_train == i]
    #     datai = datai - datai.mean(0)
    #     Data_train.T[Label_train == i] = datai

    # Data_train_ndim, eig_vec = LR_LPP(Data_train.T, ndim)
    Data_train_ndim, eig_vec = pca(Data_train.T, ndim)
    # Data_train_ndim, eig_vec = PCA_L1(Data_train, ndim)
    # Data_train_ndim, eig_vec = alpp(Data_train.T,
    #                                 lam,
    #                                 n_dims=ndim,
    #                                 n_neighbors=k)
    # Data_train_ndim, eig_vec = nlpp(Data_train.T,
    #                                 lam,
    #                                 n_dims=ndim,
    #                                 n_neighbors=k,
    #                                 maxN=MaxN,
    #                                 beta=beta)
    # eig_vec = LDA_L1(Data_train.T, Label_train, ndim, 0.001)
    # Data_train_ndim = np.dot(Data_train.T, eig_vec)

    # Data_train_ndim, eig_vec = LRE(Data_train, n_dims=ndim)
    # Data_train_ndim, eig_vec = lda(Data_train.T, Label_train, n_dim=ndim)
    # Data_train_ndim, eig_vec = nlpp_n(Data_train.T,
    #                                   lam,
    #                                   n_dims=ndim,
    #                                   n_neighbors=k,
    #                                   maxN=MaxN,
    #                                   beta=beta)
    # dist = cal_pairwise_dist(Data_train)
    # max_dist = np.max(dist)
    # Data_train_ndim, eig_vec = lpp(Data_train.T,
    #                                n_dims=ndim,
    #                                n_neighbors=k,
    #                                t=lam * max_dist)

    Data_test = Data_create(path_Test)
    # Data_test = min_max_scaler.fit_transform(Data_test)
    Label_orig = np.zeros(Data_test.shape[1])
    # Data_test = Data_test.astype(np.float32)
    for i in range(Label_orig.size):
        Label_orig[i] = int(i / test_num)
    # clusters = np.unique(Label_orig)
    # for i in clusters:
    #     datai = Data_test.T[Label_orig == i]
    #     datai = datai - datai.mean(0)
    #     Data_test.T[Label_orig == i] = datai
    # Data_test = Data_test - np.mean(Data_test, keepdims=True)
    Data_test_ndim = np.dot(Data_test.T, eig_vec)

    clf = svm.SVC(kernel='linear')
    clf.fit(Data_train_ndim, Label_train)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(Data_train_ndim, Label_train)
    Label_predict_n = neigh.predict(Data_test_ndim)
    Label_predict = KNN(Data_train_ndim, Label_train, Data_test_ndim, 1)
    Label_predict = np.array(Label_predict)
    Label_predict_svm = clf.predict(Data_test_ndim)
    accuracy_cos = np.mean(Label_predict == Label_orig) * 100
    accuracy_euc = np.mean(Label_predict_n == Label_orig) * 100
    accuracy_svm = np.mean(Label_predict_svm == Label_orig) * 100
    # print('cos:' + str(accuracy_cos), 'euc:' + str(accuracy_euc),
    #       'svm:' + str(accuracy_svm))
    return accuracy_cos, accuracy_euc, accuracy_svm


def data_Analyze1(lam, path1, path2, ndim, k, train_num, test_num, beta, MaxN):

    path_Train, path_Test = Buid_Train_Test(path1, path2, train_num)
    Data_train = Data_create(path_Train)
    # Data_train = preprocessing.scale(Data_train)
    Label_train = np.zeros(Data_train.shape[1])
    # Data_train = Data_train.astype(np.float32)
    for i in range(Label_train.size):
        Label_train[i] = int(i / train_num)

    # clusters = np.unique(Label_train)
    # for i in clusters:
    #     datai = Data_train.T[Label_train == i]
    #     datai = datai - datai.mean(0)
    #     Data_train.T[Label_train == i] = datai

    # Data_train_ndim, eig_vec = LR_LPP(Data_train.T, ndim)
    # Data_train_ndim, eig_vec = pca(Data_train.T, ndim)
    # Data_train_ndim, eig_vec = PCA_L1(Data_train, ndim)
    # Data_train_ndim, eig_vec = alpp(Data_train.T,
    #                                 lam,
    #                                 n_dims=ndim,
    #                                 n_neighbors=k)
    # Data_train_ndim, eig_vec = nlpp(Data_train.T,
    #                                 lam,
    #                                 n_dims=ndim,
    #                                 n_neighbors=k,
    #                                 maxN=MaxN,
    #                                 beta=beta)
    # eig_vec = LDA_L1(Data_train.T, Label_train, ndim, 0.001)
    # Data_train_ndim = np.dot(Data_train.T, eig_vec)

    # Data_train_ndim, eig_vec = LRE(Data_train, n_dims=ndim)
    # Data_train_ndim, eig_vec = lda(Data_train.T, Label_train, n_dim=ndim)
    # Data_train_ndim, eig_vec = nlpp_n(Data_train.T,
    #                                   lam,
    #                                   n_dims=ndim,
    #                                   n_neighbors=k,
    #                                   maxN=MaxN,
    #                                   beta=beta)
    dist = cal_pairwise_dist(Data_train)
    max_dist = np.max(dist)
    Data_train_ndim, eig_vec = lpp(Data_train.T,
                                   n_dims=ndim,
                                   n_neighbors=k,
                                   t=lam * max_dist)

    Data_test = Data_create(path_Test)
    # Data_test = preprocessing.scale(Data_test)
    Label_orig = np.zeros(Data_test.shape[1])
    # Data_test = Data_test.astype(np.float32)
    for i in range(Label_orig.size):
        Label_orig[i] = int(i / test_num)
    # clusters = np.unique(Label_orig)
    # for i in clusters:
    #     datai = Data_test.T[Label_orig == i]
    #     datai = datai - datai.mean(0)
    #     Data_test.T[Label_orig == i] = datai
    Data_test = Data_test - np.mean(Data_test, keepdims=True)
    Data_test_ndim = np.dot(Data_test.T, eig_vec)

    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    ppn.fit(Data_train_ndim, Label_train)

    clf = svm.SVC(kernel='linear')
    clf.fit(Data_train_ndim, Label_train)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(Data_train_ndim, Label_train)
    Label_predict_n = neigh.predict(Data_test_ndim)
    Label_predict = KNN(Data_train_ndim, Label_train, Data_test_ndim, 1)
    Label_predict = np.array(Label_predict)
    Label_predict_svm = clf.predict(Data_test_ndim)
    Label_predict_linear = ppn.predict(Data_test_ndim)
    accuracy_cos = np.mean(Label_predict == Label_orig) * 100
    accuracy_euc = np.mean(Label_predict_n == Label_orig) * 100
    accuracy_svm = np.mean(Label_predict_svm == Label_orig) * 100
    accuracy_lin = np.mean(Label_predict_linear == Label_orig) * 100
    # print('cos:' + str(accuracy_cos), 'euc:' + str(accuracy_euc),
    #       'svm:' + str(accuracy_svm))
    return accuracy_cos, accuracy_euc, accuracy_svm, accuracy_lin


if __name__ == '__main__':
    s1 = 0
    s2 = 0
    s3 = 0
    img_path = 'F:/Database/Traffic'
    analy_path = 'F:/Database_test/Traffic_test'
    total_num = 20
    train_num = 10
    test_num = total_num - train_num
    lam = 0.1
    maxn = 20
    k_n = 20
    s1_n = 0
    s2_n = 0
    s3_n = 0
    # for i in range(3):
    #     s1, s2, s3 = data_Analyze1(lam,
    #                                img_path,
    #                                analy_path,
    #                                ndim=100,
    #                                k=5,
    #                                train_num=train_num,
    #                                test_num=test_num,
    #                                beta=1.5,
    #                                MaxN=20)
    #     s1_n += s1
    #     s2_n += s2
    #     s3_n += s3
    # print('cos:' + str(s1_n / 3), 'euc:' + str(s2_n / 3),
    #       'svm:' + str(s3_n / 3))
    s1, s2, s3, s4 = data_Analyze1(lam,
                                   img_path,
                                   analy_path,
                                   ndim=100,
                                   k=5,
                                   train_num=train_num,
                                   test_num=test_num,
                                   beta=1.5,
                                   MaxN=maxn)
    print('cos:' + str(s1), 'euc:' + str(s2), 'svm:' + str(s3),
          'lin:' + str(s4))
    # t = 1
    # while k_n <= 18:
    #     s1_n = 0
    #     s2_n = 0
    #     s3_n = 0
    #     for i in range(t):
    #         s1, s2, s3 = data_Analyze(k_n,
    #                                   lam,
    #                                   img_path,
    #                                   analy_path,
    #                                   ndim=100,
    #                                   k=5,
    #                                   train_num=train_num,
    #                                   test_num=test_num,
    #                                   beta=1.5,
    #                                   MaxN=maxn)
    #         s1_n += s1
    #         s2_n += s2
    #         s3_n += s3
    #     print('cos:' + str(s1_n / t), 'euc:' + str(s2_n / t),
    #           'svm:' + str(s3_n / t))
    #     #     train_num += 1
    #     #     test_num -= 1
    #     k_n += 2

    # while maxn <= 10:
    #     s1_n = 0
    #     s2_n = 0
    #     s3_n = 0
    #     for i in range(10):
    #         s1, s2, s3 = data_Analyze(lam,
    #                                   img_path,
    #                                   analy_path,
    #                                   ndim=100,
    #                                   k=5,
    #                                   train_num=train_num,
    #                                   test_num=test_num,
    #                                   beta=1.5,
    #                                   MaxN=20)
    #         s1_n += s1
    #         s2_n += s2
    #         s3_n += s3
    #     print('cos:' + str(s1_n / 10), 'euc:' + str(s2_n / 10),
    #           'svm:' + str(s3_n / 10))
    #     maxn += 1
    # s = 0
    # s_n = 0
    # lam = 0.1
    # b = 0.01
    # beta = 1.5
    # while (beta < 1.6):
    #     for i in range(3):
    #         s += data_Analyze(lam,
    #                           img_path,
    #                           analy_path,
    #                           ndim=100,
    #                           k=5,
    #                           train_num=train_num,
    #                           test_num=test_num,
    #                           beta=beta)
    #     if ((s / 3) > s_n):
    #         lam_f = beta
    #         s_n = s / 3
    #     s = 0
    #     beta += b
    # print('final num:', lam_f)
