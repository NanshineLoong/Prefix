import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from data.generate_dataset import Dataset

model_path = "./checkpoint/rfc.pkl"

class Exp_random_forest:
    def __init__(self):
        pass

    def load_dataset(self, mode='train'):
        dataset = Dataset()
        if mode == 'train':
            x_data, label_data = dataset.load_data(mode='train')
            return train_test_split(x_data, label_data, test_size=0.1, random_state=0, stratify=label_data)
        elif mode == 'test':
            x_data, label_data = dataset.load_data(mode='test')
            return x_data, label_data

        # 截取部分特征
        # x_data = x_data[:, 1:270]
        # print(x_data.shape)

    def train(self):
        x_train, x_test, y_train, y_test = self.load_dataset(mode='train')
        # 网格化搜索最优参数
        # param_test = {'max_features': range(3, 11, 2)}
        # g_search = GridSearchCV(estimator=RandomForestClassifier(n_estimators=140, random_state=12345),
        #                         param_grid=param_test, scoring='f1', cv=5)
        # g_search.fit(x_train, y_train)
        # print(g_search.best_params_)

        rfc = RandomForestClassifier(n_estimators=170, random_state=12345)
        print("------------------------------")
        print("training...")
        rfc.fit(x_train, y_train)
        print(rfc.max_depth)
        pred = rfc.predict(x_test)
        acc = accuracy_score(y_test, pred)
        print(acc)

        p = precision_score(y_test, pred)
        r = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(p, r, f1)

        pickle.dump(rfc, open(model_path, 'wb'))
        print("------------------------------")
        # 计算混淆矩阵
        # confusion_matrix(y_test, pred)
        # print("confusion matrixs:", confusion_matrix)

    def test(self):
        x_test, y_test = self.load_dataset(mode='test')
        rfc = pickle.load(open(model_path, 'rb'))
        print("------------------------------")
        print("testing...")
        pred = rfc.predict(x_test)
        acc = accuracy_score(y_test, pred)
        print(acc)

        p = precision_score(y_test, pred)
        r = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(p, r, f1)
        print("------------------------------")


