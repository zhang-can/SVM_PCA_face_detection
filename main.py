# -*- coding: utf-8 -*-
"""
Created on 20171130

@author:  ZhangCan
@github:  zhang-can
@project: https://github.com/zhang-can/SVM_PCA_face_detection
"""
from time import time
from PIL import Image
import glob
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
#import matplotlib.pyplot as plt
import math

# 配置AR数据集路径
PICTURE_PATH = u"./dataset/AR/"

# 读取所有图片并一维化
def get_picture():
    label = 1
    while (label <= 120):
        sub_label = 1
        while(sub_label <= 26):
            file_name = PICTURE_PATH + "\\AR" + str(label).zfill(3) + "-" + str(sub_label) + ".tif"
            for name in glob.glob(file_name):
                img = Image.open(name)
                all_data_set.append(list(img.getdata()))
                all_data_label.append(label)
                sub_label += 1
        label += 1

# 输入核函数名称和参数gamma值，返回SVM训练十折交叉验证的准确率
def SVM(kernel_name, param):
    # 十折交叉验证计算出平均准确率
    # n_splits交叉验证，随机取
    kf = KFold(n_splits=10, shuffle=True)
    precision_average = 0.0
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma=param),
                       param_grid)
    for train, test in kf.split(X):
        clf = clf.fit(X[train], y[train])
        # print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        # print classification_report(y[test], test_pred)
        # 计算平均准确率
        precision = 0
        for i in range(0, len(y[test])):
            if (y[test][i] == test_pred[i]):
                precision = precision + 1
        precision_average = precision_average + float(precision) / len(y[test])
    precision_average = precision_average / 10

    return precision_average


# N重交叉验证计算出平均准确率
def N_fold(clf, fold_size):

    precision_average = 0.0
    kf = KFold(n_splits=fold_size, shuffle=True)

    test_k = []
    train_k = []

    # 用于展示错误识别图片
    show_pic_num = []

    j = 0
    for i in range(0, 120):
        for train, test in kf.split(np.arange(26)):
            test_offset = np.linspace(26 * i, 26 * i, len(test))
            test_k.append(test + test_offset)
            train_offset = np.linspace(26 * i, 26 * i, len(train))
            train_k.append(train + train_offset)

    for k in range(0, fold_size):
        test_key = np.array([])
        train_key = np.array([])
        for i in range(0, 120):
            test_key = np.append(test_key, test_k[k + fold_size * i])
            train_key = np.append(train_key, train_k[k + fold_size * i])

        clf = clf.fit(X[train_key.astype(np.int32)], y[train_key.astype(np.int32)])
        test_pred = clf.predict(X[test_key.astype(np.int32)])
        #print(classification_report(y[test_key.astype(np.int32)], test_pred))
        precision = 0
        for i in range(0, len(y[test_key.astype(np.int32)])):
            if (y[test_key.astype(np.int32)][i] == test_pred[i]):
                precision = precision + 1
                #print(X[test_key.astype(np.int32)].shape)
            else:
                # 输出错误分类信息
                print("true label: ", y[test_key.astype(np.int32)][i], "wrong label: ", test_pred[i])
                true_pic_num = test_key.astype(np.int32)[i]
                true_pic_offset = (test_key.astype(np.int32)[i]+1)%26
                wrong_pic_num = test_pred[i] * 26 + true_pic_offset - 1
                show_pic_num.append(true_pic_num)
                show_pic_num.append(wrong_pic_num)
                '''
                # 单组输出错误识别图片
                plt.figure(5)
                plt.subplot(1, 2, 1).imshow(np.array(all_data_set)[true_pic_num].reshape((100, 80)),cmap="gray")
                plt.title("Test image: ")
                plt.subplot(1, 2, 2).imshow(np.array(all_data_set)[wrong_pic_num].reshape((100, 80)),cmap="gray")
                plt.title("Error identified as: ")
                plt.show()
                '''

        single_precision = float(precision) / len(y[test_key.astype(np.int32)])
        precision_average = precision_average + single_precision
    precision_average = precision_average/fold_size

    '''
    # 显示识别错误图片对比
    figure_h = math.ceil((len(show_pic_num)/2)**0.5)
    figure_w = figure_h * 2

    plt.figure("Show Wrongly identified images")
    for k in range(0, math.ceil(len(show_pic_num)/2)):
        plt.subplot(figure_h, figure_w, 2*k+1).imshow(np.array(all_data_set)[show_pic_num[2*k]].reshape((100, 80)), cmap="gray")
        plt.title("Test image: ")

        plt.subplot(figure_h, figure_w, 2*k+2).imshow(np.array(all_data_set)[show_pic_num[2*k+1]].reshape((100, 80)), cmap="gray")
        plt.title("Error identified as: ")

    plt.xticks([])
    plt.yticks([])
    plt.show()
    '''
    return precision_average

all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签
get_picture()


'''
# 输出核函数与gamma测试图
t0 = time()
n_components = 80
pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
# X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)
kernel_to_test = ['rbf', 'poly', 'sigmoid']
# rint SVM(kernel_to_test[0], 0.1)
plt.figure(1)

for kernel_name in kernel_to_test:
    x_label = np.linspace(0.0001, 1, 100)
    y_label = []
    for i in x_label:
        y_label.append(SVM(kernel_name, i))
    plt.plot(x_label, y_label, label=kernel_name)

print("done in %0.3fs" % (time() - t0))
plt.xlabel("Gamma")
plt.ylabel("Precision")
plt.title('Different Kernels Contrust')
plt.legend()
plt.show()

'''

'''
# n_components测试(3-1)
plt.figure(4)
x_label = range(50, 58)
y_label = []
n_components_test_result={}
'''
for n_components in range(80,81):
    # PCA降维
    pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(all_data_set)
    # PCA降维后的总数据集
    all_data_pca = pca.transform(all_data_set)
    eigenfaces = pca.components_.reshape((n_components, 100, 80))
    # X为降维后的数据，y是对应类标签
    X = np.array(all_data_pca)
    y = np.array(all_data_label)

    '''
    # 输出Eigenfaces
    plt.figure("Eigenfaces")
    for i in range(1, 81):
        plt.subplot(8, 10, i).imshow(eigenfaces[i-1], cmap="gray")
        plt.xticks(())
        plt.yticks(())

    plt.show()
    '''

    t0 = time()
    param_grid = {'C': [100],
                   'gamma': [0.01], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    print("\nn_components: " + str(n_components))

    '''
    # 输出不同fold_size准确率变化曲线
    plt.figure(1)
    x_label = []
    y_label = []
    for m in range(2,11):
        acc = 0
        x_label.append(m)
        for n in range(0, 10):
            acc = acc + N_fold(clf, m)*100
        y_label.append(acc/10)
        plt.plot(x_label, y_label)

    print("done in %0.3fs" % (time() - t0))
    plt.xlabel("fold_size")
    plt.ylabel("Precision")
    plt.title('Different fold_size')
    plt.legend()
    plt.show()
    '''

    fold_size = 5
    accuracy = 0
    for n in range(0, 10):
        accuracy = accuracy + N_fold(clf, fold_size) * 100
    print("\n-------------------------------------------")
    print("10次", fold_size, "重交叉平均准确率为" + str(accuracy / 10) + "%")
    print("-------------------------------------------\n")

    print("done in %0.3fs" % (time() - t0))
    '''
    # n_components测试(3-2)
    n_components_test_result[str(n_components)] = str(accuracy/10)
    y_label.append(accuracy)
    '''

'''
# n_components测试(3-3)
print(n_components_test_result)
plt.plot(x_label, y_label)

plt.xlabel("n_components")
plt.ylabel("Precision")
plt.title('n_components_test_result')
plt.legend()
plt.show()
'''
