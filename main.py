import os

import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def data_select(datas, labels, label1, label2):
    """选择标签为label1和label2的数据，服务于one-vs-one"""
    data_selected = []
    label_selected = []
    for i in range(len(datas)):
        if labels[i] == label1 or labels[i] == label2:
            data_selected.append(datas[i])
            label_selected.append(labels[i])
    return data_selected, label_selected


def data_change(datas, labels, label):
    """将标签不是label的数据标签标为-1，服务于one-vs-rest"""
    data_changed = []
    label_changed = []
    for i in range(len(datas)):
        data_changed.append(datas[i])
        if labels[i] != label:
            label_changed.append(-1)
        else:
            label_changed.append(labels[i])
    return data_changed, label_changed


class SvmForOneVsOne:
    """one-vs-one模型"""

    def __init__(self, class_num, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False, random_state=None):
        """初始化模型"""
        self.decision_function_shape = decision_function_shape
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.tol = tol
        self.probability = probability
        self.shrinking = shrinking
        self.coef0 = coef0
        self.class_num = class_num
        self.SVM_num = int(self.class_num * (self.class_num - 1) / 2)
        self.set_of_SVM = []
        self.C = C
        self.kernel = kernel
        self.break_ties = break_ties
        # 为每一对种类创建一个SVM二分类模型
        for i in range(class_num - 1):
            SVMs = []
            j = i + 1
            while j < class_num:
                SVMs.append(SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                                shrinking=self.shrinking, probability=self.probability, tol=self.tol,
                                cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose,
                                max_iter=self.max_iter, decision_function_shape=self.decision_function_shape,
                                break_ties=self.break_ties, random_state=self.random_state))
                j = j + 1
            self.set_of_SVM.append(SVMs)

    def train(self, train_datas, train_labels):
        """训练模型"""
        for i in range(len(self.set_of_SVM)):
            for j in range(len(self.set_of_SVM[i])):
                label1 = i
                label2 = i + j + 1
                # 使用对应的数据对相应的模型进行训练
                train_data, train_label = data_select(train_datas, train_labels, label1, label2)
                self.set_of_SVM[i][j].fit(np.array(train_data), np.array(train_label))

    def predict(self, data):
        """预测数据所属类型"""
        classes = np.zeros(self.class_num)
        # 每个SVM二分类模型对测试数据进行预测，并进行投票
        for i in range(len(self.set_of_SVM)):
            for j in range(len(self.set_of_SVM[i])):
                predict_class = self.set_of_SVM[i][j].predict(data)
                classes[predict_class] = classes[predict_class] + 1
        # 返回得票最多的种类作为测试数据的预测种类
        return classes.argmax()


class SvmForOneVsRest:
    """one-vs-rest模型"""

    def __init__(self, class_num, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                 probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False, random_state=None):
        """初始化模型"""
        self.decision_function_shape = decision_function_shape
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.tol = tol
        self.probability = probability
        self.shrinking = shrinking
        self.coef0 = coef0
        self.class_num = class_num
        self.SVM_num = class_num
        self.set_of_SVM = []
        self.C = C
        self.kernel = kernel
        self.break_ties = break_ties
        # 为每一个种类创建一个SVM二分类模型
        for i in range(class_num):
            self.set_of_SVM.append(
                SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                    shrinking=self.shrinking, probability=self.probability, tol=self.tol,
                    cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose,
                    max_iter=self.max_iter, decision_function_shape=self.decision_function_shape,
                    break_ties=self.break_ties, random_state=self.random_state))

    def train(self, train_datas, train_labels):
        """训练模型"""
        for i in range(len(self.set_of_SVM)):
            label = i
            # 使用改变好标签的数据对相应的模型进行训练
            train_data, train_label = data_change(train_datas, train_labels, label)
            self.set_of_SVM[i].fit(np.array(train_data), np.array(train_label))

    def predict(self, data):
        """预测数据所属类型"""
        max_i = 0
        max_proba = 0
        # 每个SVM二分类模型对测试数据进行预测，得到测试数据所属某个种类的概率
        for i in range(len(self.set_of_SVM)):
            proba = self.set_of_SVM[i].predict_proba(data)
            if proba[0][1] > max_proba:
                max_i = i
                max_proba = proba[0][1]
        # 返回概率最大的种类
        return max_i


def test(SVM, test_data, test_label):
    """用测试集测试某个SVM，并返回平均准确率"""
    total_num = len(test_data)
    right = 0
    predicts = []
    for i in range(total_num):
        predicts.append(SVM.predict([test_data[i]]))
        if SVM.predict([test_data[i]]) == test_label[i]:
            right = right + 1
    acc = right / total_num
    return acc


def SubjectDependencyOneVsOne(class_num, C, train_datas, train_labels, test_datas, test_labels):
    """被试依存one-vs-one训练测试模板"""
    SVMs = []
    # 训练部分
    for i in range(len(train_datas)):
        SVMs.append(SvmForOneVsOne(class_num, C=C))
        SVMs[i].train(train_datas[i], train_labels[i])
    accs = []
    # 测试部分
    for i in range(len(test_datas)):
        acc = test(SVMs[i], test_datas[i], test_labels[i])
        accs.append(acc)
    acc_avg = np.array(accs).sum() / len(accs)
    return acc_avg


def SubjectDependencyOneVsRest(class_num, C, train_datas, train_labels, test_datas, test_labels):
    """被试依存one-vs-rest训练测试模板"""
    SVMs = []
    # 训练部分
    for i in range(len(train_datas)):
        SVMs.append(SvmForOneVsRest(class_num, C=C))
        SVMs[i].train(train_datas[i], train_labels[i])
    accs = []
    # 测试部分
    for i in range(len(test_datas)):
        acc = test(SVMs[i], test_datas[i], test_labels[i])
        accs.append(acc)
    acc_avg = np.array(accs).sum() / len(accs)
    return acc_avg


def data_to_people(people_num, datas, labels):
    """拼接数据，获取每个人对应的数据，用于被试独立"""
    peoples_data_list_version = []
    peoples_label_list_version = []
    for i in range(len(datas)):
        if i < people_num:
            peoples_data_list_version.append([datas[i]])
            peoples_label_list_version.append([labels[i]])
        else:
            peoples_data_list_version[i % 15].append(datas[i])
            peoples_label_list_version[i % 15].append(labels[i])
    peoples_data = []
    peoples_label = []
    for i in range(people_num):
        people_data = []
        people_label = []
        for j in range(len(peoples_data_list_version[i])):
            for k in range(len(peoples_data_list_version[i][j])):
                people_data.append(peoples_data_list_version[i][j][k])
                people_label.append(peoples_label_list_version[i][j][k])
        peoples_data.append(people_data)
        peoples_label.append(people_label)
    return peoples_data, peoples_label


def get_datas(test_num, peoples_datas, peoples_labels):
    """根据轮数，获取对应轮次的训练集和测试集，用于被试独立"""
    train_datas = []
    train_labels = []
    for people in range(len(peoples_datas)):
        if people == test_num:
            test_datas = [peoples_datas[people]]
            test_labels = [peoples_labels[people]]
        else:
            for i in range(len(peoples_datas[people])):
                train_datas.append(peoples_datas[people][i])
                train_labels.append(peoples_labels[people][i])
    return [train_datas], [train_labels], test_datas, test_labels


if __name__ == '__main__':
    time_start = time.time()
    Cs = []
    order = []
    for i in range(100):
        Cs.append((i + 1) / 10)
    accs = []
    C_for_independent = 1
    dir_name = './SEED-IV'
    sessions = os.listdir(dir_name)
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    people_num = 15
    class_num = 4
    # 读取文件夹中的所有数据
    for session in sessions:
        session_dir = dir_name + '/' + session
        peoples = os.listdir(session_dir)
        for people in peoples:
            people_dir = session_dir + '/' + people + '/'
            train_data = np.load(people_dir + 'train_data.npy')
            train_label = np.load(people_dir + 'train_label.npy')
            test_data = np.load(people_dir + 'test_data.npy')
            test_label = np.load(people_dir + 'test_label.npy')
            train_data = train_data.reshape((train_data.shape[0], -1))
            test_data = test_data.reshape((test_data.shape[0], -1))
            train_datas.append(train_data)
            train_labels.append(train_label)
            test_datas.append(test_data)
            test_labels.append(test_label)
    print("0:被试依存(one vs one)")
    print("1:被试依存(one vs rest)")
    print("2:被试独立(one vs one)")
    print("3:被试独立(one vs rest)")
    '''SVM = SVC()
    SVM.fit(train_datas[1], train_labels[1])
    acc = test(SVM, test_datas[1], test_labels[1])
    print(acc)'''
    selection = input()
    if selection == '0':
        # 被试依存(one vs one)
        for C in tqdm(Cs):
            acc = SubjectDependencyOneVsOne(class_num, C, train_datas, train_labels, test_datas, test_labels)
            accs.append(acc)
        plt.plot(Cs, accs)
        plt.show()
    if selection == '1':
        # 被试依存(one vs rest)
        for C in tqdm(Cs):
            acc = SubjectDependencyOneVsRest(class_num, C, train_datas, train_labels, test_datas, test_labels)
            accs.append(acc)
        plt.plot(Cs, accs)
        plt.show()
    if selection == '2':
        # 被试独立(one vs one)
        peoples_data, peoples_label = data_to_people(people_num, train_datas + test_datas, train_labels + test_labels)
        for i in tqdm(range(people_num)):
            train_datas, train_labels, test_datas, test_labels = get_datas(i, peoples_data, peoples_label)
            acc = SubjectDependencyOneVsOne(class_num, C_for_independent, train_datas, train_labels, test_datas,
                                            test_labels)
            accs.append(acc)
        print("accs:")
        print(accs)
        acc_avg = np.array(accs).sum() / len(accs)
        print("acc_avg:" + str(acc_avg))
        time_end = time.time()
        total_time = time_end - time_start
        print("total time:")
        print(total_time)
    if selection == '3':
        # 被试独立(one vs rest)
        peoples_data, peoples_label = data_to_people(people_num, train_datas + test_datas,
                                                     train_labels + test_labels)
        for i in tqdm(range(people_num)):
            train_datas, train_labels, test_datas, test_labels = get_datas(i, peoples_data, peoples_label)
            acc = SubjectDependencyOneVsRest(class_num, C_for_independent, train_datas, train_labels, test_datas,
                                             test_labels)
            accs.append(acc)
        print("accs:")
        print(accs)
        acc_avg = np.array(accs).sum() / len(accs)
        print("acc_avg:" + str(acc_avg))
        time_end = time.time()
        total_time = time_end - time_start
        print("total time:")
        print(total_time)
