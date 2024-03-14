import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    #Generate a swiss roll dataset.
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = 83 * np.random.rand(1, n_samples)
    z = t * np.sin(t)
    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    return X, t


def rbf(dist, t):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist / t))


def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist


def cal_rbf_dist(data, n_neighbors, t):

    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W


def Get_A_it(W, D):  ##获取收敛权值矩阵A,迭代得出
    tol = 1e-15
    n = 2
    D_n = np.linalg.inv(np.sqrt(D))
    S = np.dot(D_n.dot(W), D_n)
    Tmp_A = np.zeros_like(S)
    Tmp_A = S
    while True:
        A = np.linalg.matrix_power(S, n)
        if (np.max(np.abs(A - Tmp_A)) < tol):
            break
        Tmp_A = A
        n = n + 1
        print(n)
    return A


def Get_A(W, D):  #获取收敛权值矩阵A,公式得出
    D_n = np.linalg.inv(sqrtm(D))
    S = np.dot(D_n.dot(W), D_n)
    eig_val, eig_vec = np.linalg.eig(S)
    indexs_ = np.argsort(-eig_val)
    eig_val_n = eig_val[indexs_]
    eig_vec_n = eig_vec[:, indexs_]
    i = 0
    while (True):
        if (eig_val_n[i] == eig_val_n[i + 1]):
            i = i + 1
        else:
            break
    eig_vec_picked = np.real(eig_vec_n[:, 0:i + 1])
    A = np.dot(eig_vec_picked, eig_vec_picked.T)
    return A


def Get_t(data, lam):
    dist = cal_pairwise_dist(data)
    max_dist = np.max(dist)
    # print("max_dist", max_dist)
    t = lam * max_dist
    return t


#公式法:
def alpp(data, lam, n_dims, n_neighbors):  #传入图片矩阵,最后维度,紧邻数
    t = Get_t(data, lam)
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)

    for i in range(N):
        D[i, i] = np.sum(W[i])

    A = Get_A(W, D)
    for i in range(N):
        D[i, i] = np.sum(A[i])

    L = D - A
    XDXT = np.dot(np.dot(data.T, D), data)
    XLXT = np.dot(np.dot(data.T, L), data)

    eig_val, eig_vec = np.linalg.eig(np.dot(np.linalg.pinv(XDXT), XLXT))

    sort_index_ = np.argsort(np.abs(eig_val))
    eig_val = eig_val[sort_index_]
    # print("eig_val[:10]", eig_val[:10])

    j = 0
    while eig_val[j] < 1e-6:
        j += 1

    # print("j: ", j)

    sort_index_ = sort_index_[j:j + n_dims]
    # print(sort_index_)
    eig_val_picked = eig_val[j:j + n_dims]

    eig_vec_picked = np.real(eig_vec[:, sort_index_])
    # print(eig_vec_picked)
    data_ndim = np.dot(data, eig_vec_picked)

    return data_ndim, eig_vec_picked  #输出降维后图片矩阵和特征矩阵


#迭代法:
def alpp_it(data, lam, n_dims, n_neighbors):
    t = Get_t(data, lam)
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)

    for i in range(N):
        D[i, i] = np.sum(W[i])

    A = Get_A_it(W, D)
    for i in range(N):
        D[i, i] = np.sum(A[i])

    L = D - A
    XDXT = np.dot(np.dot(data.T, D), data)
    XLXT = np.dot(np.dot(data.T, L), data)

    eig_val, eig_vec = np.linalg.eig(np.dot(np.linalg.pinv(XDXT), XLXT))

    sort_index_ = np.argsort(np.abs(eig_val))
    eig_val = eig_val[sort_index_]
    # print("eig_val[:10]", eig_val[:10])

    j = 0
    while eig_val[j] < 1e-6:
        j += 1

    # print("j: ", j)

    sort_index_ = sort_index_[j:j + n_dims]
    # print(sort_index_)
    eig_val_picked = eig_val[j:j + n_dims]

    eig_vec_picked = np.real(eig_vec[:, sort_index_])
    # print(eig_vec_picked)
    data_ndim = np.dot(data, eig_vec_picked)

    return data_ndim, eig_vec_picked


if __name__ == "__main__":
    X, y = make_swiss_roll(n_samples=1000)
    data_2d, w = alpp(X, lam=0.01, n_dims=2, n_neighbors=5)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("aLPP")
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y)
    plt.show()
