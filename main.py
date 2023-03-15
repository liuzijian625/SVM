import os

import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm


def data_select(datas, labels, label1, label2):
    data_selected = []
    label_selected = []
    for i in range(len(datas)):
        if labels[i] == label1 or labels[i] == label2:
            data_selected.append(datas[i])
            label_selected.append(labels[i])
    return data_selected, label_selected


class SvmForOneVsOne:
    def __init__(self, class_num, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False, random_state=None):
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
        for i in range(len(self.set_of_SVM)):
            for j in range(len(self.set_of_SVM[i])):
                label1 = i
                label2 = i + j + 1
                train_data, train_label = data_select(train_datas, train_labels, label1, label2)
                self.set_of_SVM[i][j].fit(np.array(train_data), np.array(train_label))

    def predict(self, data):
        classes = np.zeros(4)
        for i in range(len(self.set_of_SVM)):
            for j in range(len(self.set_of_SVM[i])):
                predict_class = self.set_of_SVM[i][j].predict([data])
                classes[predict_class] = classes[predict_class] + 1
        return classes.argmax()


class SvmForOneVsRest:
    def __init__(self, class_num):
        self.class_num = class_num
        self.SVM_num = self.class_num


def test(SVM, test_data, test_label):
    total_num = len(test_data)
    right = 0
    for i in range(total_num):
        if SVM.predict(test_data[i]) == test_label[i]:
            right = right + 1
    acc = right / total_num
    return acc


if __name__ == '__main__':
    dir_name = './SEED-IV'
    sessions = os.listdir(dir_name)
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    class_num = 4
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
    selection = input()
    if selection == '0':
        print('开始训练')
        SVMs = []
        for i in tqdm(range(len(train_datas))):
            SVMs.append(SvmForOneVsOne(class_num))
            SVMs[i].train(train_datas[i], train_labels[i])
        print('训练完成')
        print('测试开始')
        accs = []
        for i in tqdm(range(len(test_datas))):
            acc = test(SVMs[i], test_datas[i], test_labels[i])
            accs.append(acc)
        acc_avg = np.array(accs).sum() / len(accs)
        print(accs)
        print('平均准确率为：' + str(acc_avg * 100) + '%')
