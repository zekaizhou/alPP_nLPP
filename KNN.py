import numpy as np


def KNN(X_train, y_train, X_test, k):
    #修改列表为numpy类型
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    #获得训练、测试数据的长度
    X_train_len = len(X_train)
    X_test_len = len(X_test)
    pre_lable = []  #存储预测标签
    '''
    依次遍历测试数据，计算每个测试数据与训练数据的距离值，排序，
    根据前K个投票选举出预测结果
    '''
    for test_len in range(X_test_len):  #计算测试的第一组数据
        dis = []
        for train_len in range(X_train_len):
            #计算距离
            # dis_EUC = np.linalg.norm(X_train[train_len, :] -
            #                          X_test[test_len, :])
            # dis.append(dis_EUC)
            s = np.dot(X_train[train_len, :], X_test[test_len, :].T)
            u = (np.linalg.norm(X_train[train_len, :]) *
                 np.linalg.norm(X_test[test_len, :]))
            if u == 0:
                dis_COS = 0
            else:
                dis_COS = -(s / u)
            dis.append(dis_COS)

        dis = np.array(dis)
        sort_id = dis.argsort()
        # 按照升序进行快速排序，返回的是原数组的下标。
        # 比如，x = [30, 10, 20, 40]
        # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
        # 那么，numpy.argsort(x) = [1, 2, 0, 3]

        dic = {}
        for i in range(k):
            vlable = y_train[sort_id[i]]  #为对应的标签记数
            dic[vlable] = dic.get(vlable, 0) + 1
            #寻找vlable代表的标签，如果没有返回0并加一，如果已经存在返回改键值对应的值并加一
        max = 0
        for index, v in dic.items():  #.items  返回所有的键值对
            if v > max:
                max = v
                maxIndex = index
        pre_lable.append(maxIndex)
    return pre_lable


if __name__ == "__main__":
    '''
    X_train=[[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12]]
    y_train=[1,2,3]
    X_test=[[1,2,3,4],   #那么预测数据应为 1、2
            [5,6,7,8]]
    '''

    X_train = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 2, 12], [1, 7, 3, 4],
               [5, 9, 7, 8], [9, 15, 11, 12]]

    y_train = [1, 2, 3, 4, 5, 6]

    X_test = [
        [9, 13, 14, 12],  # 那么预测数据应为 1、2
        [5, 6, 7, 8]
    ]
    KNN(X_train, y_train, X_test, 1)
