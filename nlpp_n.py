# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
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


def Get_t(data, lam):
    dist = cal_pairwise_dist(data)
    max_dist = np.max(dist)
    # print("max_dist", max_dist)
    t = lam * max_dist
    return t


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


def Init(data, lam, n_neighbors, n_dims):  #初始化,得到W1
    Omega = 1
    t = Get_t(data, lam)
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)

    for i in range(N):
        D[i, i] = np.sum(W[i])

    L = D - W
    XDXT = np.dot(np.dot(data.T, D), data)
    XLXT = np.dot(np.dot(data.T, L), data)
    D_n = np.linalg.inv(np.sqrt(D))
    W = np.dot(D_n.dot(W), D_n)
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
    O_W = Omega * W
    return D, O_W, W


def Update_P(data, D, O_W, n_dims):  #更新特征矩阵P
    D_n = sqrtm(D)
    N = data.shape[0]
    W_n = np.dot(np.dot(D_n, O_W), D_n)
    D_n = np.zeros_like(D)
    for i in range(N):
        D_n[i][i] = np.sum(W_n[i])

    L = D_n - O_W
    XDXT = np.dot(np.dot(data.T, D_n), data)
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

    return data_ndim, eig_vec_picked, W_n


# def Update_Omega(D, data_ndim, n, W_t):  #更新权值
#     b = []
#     x1 = (0, 1)
#     bns = []
#     D_n = np.sqrt(D)
#     for i in range(n):
#         b.append(
#             np.trace(
#                 np.dot(np.dot(np.dot(data_ndim.T, D_n), W_t[i]),
#                        np.dot(D_n, data_ndim))))
#         bns.append(x1)
#     c = -np.array(b)
#     A_eq = np.ones(shape=(1, c.size))
#     B_eq = np.array([1])
#     res = linprog(c,
#                   None,
#                   None,
#                   A_eq,
#                   B_eq,
#                   bounds=bns,
#                   method="revised simplex")
#     Omega = res.x
#     O_W = np.zeros_like(D)
#     for i in range(n):
#         O_W = O_W + (Omega[i] * W_t[i])
#     return O_W


def Update_Omega(D, data_ndim, n, W_t, beta):  #更新权值
    b = []
    a = []
    Omega = []
    D_n = sqrtm(D)
    for i in range(n):
        b.append(
            np.trace(
                np.dot(np.dot(np.dot(data_ndim.T, D_n), W_t[i]),
                       np.dot(D_n, data_ndim))))
        b[i] = pow((1 / b[i]), 1 / (beta - 1))
    b = np.array(b)
    b_sum = b.sum()
    for i in range(n):
        a.append(b[i] / b_sum)
        Omega.append(pow(a[i], beta))
    O_W = np.zeros_like(D)
    for i in range(n):
        O_W = O_W + (Omega[i] * W_t[i])
    return O_W


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


def nlpp_n(data, lam, n_dims, n_neighbors, maxN, beta):
    tol = 1e-15
    n = 1
    D, O_W, W = Init(data, lam, n_neighbors, n_dims)
    # D_n = np.sqrt(D)
    # W = np.dot(np.dot(D_n, W), D_n)
    # W = np.linalg.matrix_power(W, 10)
    W_t = (W, )
    while True:  #交替更新,直到收敛
        data_ndim, eig_vec, W_n = Update_P(data, D, O_W, n_dims)
        n = n + 1
        if n > maxN:
            break
        # D_n = np.linalg.inv(np.sqrt(D))
        # W_n = np.linalg.matrix_power(O_W, 30)
        W_n = Get_A(W_n, D)
        Wi = (W_n, )
        W_t = W_t + Wi
        if (np.max(np.abs(W_t[n - 2] - W_t[n - 1])) < tol):
            break
        O_W = Update_Omega(D, data_ndim, n, W_t, beta)

    return data_ndim, eig_vec


if __name__ == "__main__":
    X, y = make_swiss_roll(n_samples=1000)
    data_2d, w = nlpp_n(X,
                        lam=0.01,
                        n_dims=2,
                        n_neighbors=5,
                        maxN=20,
                        beta=1.5)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("aLPP")
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y)
    plt.show()
