import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import *
from random import sample
from utils.frequency import frequency_extraction
from utils.seasonality import seasonality_extraction
from utils.sequence import sequence_extraction
from utils.surge import surge_extraction

original_data_dir = './data/Prefix/'
newPrefix_data_dir = './data/newPrefix/'
split_dataset_dir = './data/splitPrefix'
feature_data_dir = './data/features/'
workdirs = ["M1", "M2", "M3"]
seasonality_files = ["0.csv"]


class Dataset:
    def __init__(self, log_templates=165, time_bin=900, duration=120, length=2*60*60, begin=30*60+2*60*60, end=24 * 60 * 60 + 30 * 60+2*60*60, theta=5, drop=True):  # 此处269是M1数据集中template的数量，使用其他数据集需要修改
        """
        :param log_templates: template的数量
        :param time bin: time bin间隔时长
        :param duration: surge特征中小时间片长度
        :param begin: failure检查的起始点
        :param end: failure检查的终止点
        :param theta: 一个有效的time bin中至少应有的template数量
        :param drop: 是否忽略failure出现在[0, 2小时 + 30分钟]时间段的time bin
        """
        self.log_templates = log_templates
        self.time_bin = time_bin
        self.duration = duration
        self.length = length
        self.begin = begin
        self.end = end
        self.theta = theta
        self.drop = drop

        self.seasonality_features = np.array(pd.DataFrame(columns=range(self.log_templates)))
    
    def load_data(self, mode='train'):
        # 没有特征提取后的文件夹，需要做特征提取
        if not os.path.exists(feature_data_dir):
            self.feature_extraction()
        else:
            print("You have extracted features!")


        
        print("loading data...")
        # dataset = pd.DataFrame(columns=['feature_vector', 'label'])
        # data_file_list = os.listdir(feature_data_dir)
        # train_data = []
        # # 合并所有数据集
        # for data_file in data_file_list:
        #     features = pd.read_csv(open(feature_data_dir + data_file, 'r+', encoding='utf-8'))
        #     dataset = dataset.append(features)
        #
        # # 去除重复项
        # dataset = dataset.drop_duplicates(subset=['feature_vector'], keep='first')

        features = []
        dataset = pd.DataFrame(columns=['feature_vector', 'label'])
        if mode == 'train':
            dataset = pd.read_csv(open(feature_data_dir + 'train.csv', 'r+', encoding='utf-8'))
        elif mode == 'test':
            dataset = pd.read_csv(open(feature_data_dir + 'test.csv', 'r+', encoding='utf-8'))

        dataset['feature_vector'] = dataset['feature_vector'].apply(
            lambda x: list(map(float, x.strip().split())))

        dataset['label'] = dataset['label'].astype('int64')
        for data in dataset['feature_vector']:
            features.append(data)
        
        print("load data successfully!")

        return np.array(features), np.array(dataset['label'])

    def generate_datasets(self):
        # for workdir in workdirs:
        #     print(f"generating {workdir}...")
        #     self.generate_dataset(workdir)
        print(f"generating newprefix...")
        self.generate_dataset(workdirs[0])

    def get_failure_list(self, pre, path):
        failures = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if pre == line.split()[0]:
                    failures.append(int(line.split()[1]))
        return failures

    def generate_dataset(self, workdir):
        # CPATH = newPrefix_data_dir + workdir # 生成的.csv文件所在路径
        CPATH = newPrefix_data_dir # 生成的.csv文件所在路径
        if not os.path.exists(CPATH):
            os.makedirs(CPATH)

        # DPATH = original_data_dir + workdir + "/template_sequence/template_sequence/" # .dat文件所在路径
        DPATH = "./data/changed_sequence"  # .dat文件所在路径
        FPATH = original_data_dir + workdir + "/failure_info.txt" 

        # 获取dat文件的所有路径
        # files_list = []
        # for _, _, files in os.walk(DPATH):
        #     for filename in files:
        #         if '.csv' in filename:
        #             files_list.append(filename)

        files_list = os.listdir(DPATH)

        for filename in files_list:
            pre = filename.split('.')[0]

            failures = self.get_failure_list(pre, FPATH) # 获取该dat文件中所有failure所在的时刻
            
            # 读取log
            try:
                log_data = pd.read_csv(os.path.join(DPATH, filename), sep=',')  # 排除空表
            except Exception:
                continue

            log_data = log_data.sort_values(by='0')
            log_data = log_data.reset_index(drop=True)

            log_sequences = pd.DataFrame(columns=['start_time', 'end_time', 'template_ids', 'timestamps', 'label'])
            template_ids, timestamps, start_times, end_times, labels = [], [], [], [], []

            start_time = int(log_data.iloc[0, 0]) // self.time_bin * self.time_bin
            end_time = int(log_data.iloc[-1, 0])
            period_end_time = start_time + self.length
            failure_time = failures[0] if len(failures) > 0 else 0
            start_index, index, failure_time_index = 0, 0, 0

            while period_end_time < end_time:
                while log_data.iloc[index, 0] < period_end_time:
                    index += 1
                
                prob_failure_time = [start_time, start_time + self.begin, start_time + self.end]
                if self.drop and prob_failure_time[0] < failure_time < prob_failure_time[1]:
                    pass
                else:
                    if index - start_index >= self.theta:
                        
                        if failure_time > prob_failure_time[2] or failure_time == 0:
                            labels.append('0')
                        elif prob_failure_time[1] <= failure_time <= prob_failure_time[2]:
                            labels.append('1')
                        else:
                            while failure_time_index < len(failures) - 1 and failures[failure_time_index] < prob_failure_time[1]:
                                failure_time_index += 1
                            failure_time = failures[failure_time_index] if failure_time_index < len(failures) else 0
                            if prob_failure_time[1] <= failure_time <= prob_failure_time[2]:
                                labels.append('1')
                            else:
                                labels.append('0')

                        template_ids.append(' '.join(list(map(str, list(log_data.iloc[start_index:index, 1])))))
                        timestamps.append(' '.join(list(map(str, list(log_data.iloc[start_index:index, 0])))))
                        start_times.append(str(start_time))
                        end_times.append(str(period_end_time))
                
                while log_data.iloc[start_index, 0] // self.time_bin * self.time_bin == start_time:
                    start_index += 1
                
                start_time = log_data.iloc[start_index, 0] // self.time_bin * self.time_bin
                period_end_time = start_time + self.length

            # 处理最后一个时间窗口
            prob_failure_time = [start_time, start_time + self.begin, start_time + self.end]
            if self.drop and prob_failure_time[1] < failure_time < prob_failure_time[2]:
                pass
            else:
                if len(log_data) - start_index >= self.theta:
                    template_ids.append(' '.join(list(map(str, list(log_data.iloc[start_index:, 1])))))
                    timestamps.append(' '.join(list(map(str, list(log_data.iloc[start_index:, 0])))))
                    start_times.append(str(start_time))
                    end_times.append(str(period_end_time))

                    if failure_time > prob_failure_time[2] or failure_time == 0:
                        labels.append('0')
                    elif prob_failure_time[1] <= failure_time <= prob_failure_time[2]:
                        labels.append('1')
                    else:
                        while failure_time_index < len(failures) - 1 and failures[failure_time_index] < prob_failure_time[1]:
                            failure_time_index += 1
                        failure_time = failures[failure_time_index] if failure_time_index < len(failures) else 0
                        if prob_failure_time[1] <= failure_time <= prob_failure_time[2]:
                            labels.append('1')
                        else:
                            labels.append('0')

            log_sequences['template_ids'], log_sequences['timestamps'], log_sequences['start_time'], \
                log_sequences['end_time'], log_sequences['label'] = template_ids, timestamps, start_times, end_times, labels

            log_sequences.to_csv(os.path.join(CPATH, pre + '.csv'), index=False)

    def pre_feature_extraction(self):
        # if not os.path.exists(newPrefix_data_dir):
        #     self.generate_datasets()
        # else:
        #     print("You have generated datasets!")

        print("pre extract features...")

        # data_file_list = os.listdir(newPrefix_data_dir)
        print("extract seasonality features...")

        for data_file in tqdm(seasonality_files):
            file_path = os.path.join(newPrefix_data_dir, data_file)
            log_sequence_data = pd.read_csv(open(file_path, 'r+', encoding='utf-8'))

            frequency_feature = frequency_extraction(log_sequence_data, self.log_templates)
            seasonality_feature = seasonality_extraction(log_sequence_data, frequency_feature, self.log_templates,
                                                         self.time_bin).reshape((1, -1))
            self.seasonality_features = np.append(self.seasonality_features, seasonality_feature, axis=0)

        self.seasonality_features = np.min(self.seasonality_features, axis=0).astype(float)
        # 将inf值用除inf外最大的数填充
        self.seasonality_features[np.isinf(self.seasonality_features)] = -np.inf
        self.seasonality_features[np.isinf(self.seasonality_features)] = np.max(self.seasonality_features)


    def split_dataset(self):
        print("split dataset into train and test...")

        data_file_list = os.listdir(newPrefix_data_dir)

        # data_file_list = sample(data_file_list, 10) # 数据集太大，随机选取部分数据文件
        data_file_list = ['25.csv']  # 测试

        dataset = pd.DataFrame(columns=['start_time', 'end_time', 'template_ids', 'timestamps', 'label'])
        # 合并所有数据集
        for data_file in data_file_list:
            features = pd.read_csv(open(newPrefix_data_dir + data_file, 'r+', encoding='utf-8'))
            dataset = dataset.append(features)

        # 去除重复项
        dataset = dataset.drop_duplicates(subset=['start_time', 'end_time', 'template_ids', 'timestamps'], keep='first')

        # 数据集划分
        x_data = dataset.iloc[:, :-1]
        label_data = dataset.iloc[:, -1:]

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, label_data, test_size=0.3, random_state=0, stratify=label_data)

        x_train['label'] = y_train
        x_test['label'] = y_test

        return x_train, x_test

        # x_train.to_csv(split_dataset_dir + "train_data" + '.csv', index=False)
        # x_test.to_csv(split_dataset_dir + "test_data" + '.csv', index=False)


    def feature_extraction(self):
        if not os.path.exists(newPrefix_data_dir):
            self.generate_datasets()
        else:
            print("You have generated datasets!")

        # 提取指定文件的seasonality特征
        self.pre_feature_extraction()

        # 划分数据集
        train_data, test_data = self.split_dataset()

        # 导入LCS set
        lcs_data = pickle.load(open('./data/lcs_set_M1.dat', 'rb+'))
        lcs_seq = set()
        for seq in lcs_data:
            # 对 lcs set中每个序列长度进行统计分析，4 和 9 分别为 0.25 和 0.75 分位数
            if 4 <= len(seq.strip().split()) <= 9:
                lcs_seq.add(seq)

        if not os.path.exists(feature_data_dir):
            os.makedirs(feature_data_dir)

        print("extracting features of training data...")
        if True:
            feature_data = pd.DataFrame(columns=['feature_vector', 'label'])
            feature_vectors, labels = [], []

            frequency_feature = frequency_extraction(train_data, self.log_templates)
            sequence_feature = sequence_extraction(train_data, lcs_seq)
            surge_feature = surge_extraction(train_data, self.log_templates, self.duration)

            # print(frequency_feature.shape, seasonality_feature.shape, sequence_feature.shape, surge_feature.shape)
            features = np.concatenate((sequence_feature, frequency_feature * self.seasonality_features,
                                    surge_feature*self.seasonality_features), axis=1)
            # print(features.shape)

            for feature in features:
                feature_vectors.append(' '.join(list(map(str, list(feature)))))

            labels.extend(list(train_data['label']))

            feature_data['feature_vector'] = feature_vectors
            feature_data['label'] = labels

            feature_data.to_csv(feature_data_dir + 'train.csv', index=False)

        print("extracting features of testing data...")
        if True:
            feature_data = pd.DataFrame(columns=['feature_vector', 'label'])
            feature_vectors, labels = [], []

            frequency_feature = frequency_extraction(test_data, self.log_templates)
            sequence_feature = sequence_extraction(test_data, lcs_seq)
            surge_feature = surge_extraction(test_data, self.log_templates, self.duration)

            # print(frequency_feature.shape, seasonality_feature.shape, sequence_feature.shape, surge_feature.shape)
            features = np.concatenate((sequence_feature, frequency_feature * self.seasonality_features,
                                       surge_feature * self.seasonality_features), axis=1)
            # print(features.shape)

            for feature in features:
                feature_vectors.append(' '.join(list(map(str, list(feature)))))

            labels.extend(list(test_data['label']))

            feature_data['feature_vector'] = feature_vectors
            feature_data['label'] = labels

            feature_data.to_csv(feature_data_dir + 'test.csv', index=False)


        # data_file_list = os.listdir(newPrefix_data_dir)
        # output_file_list = os.listdir(feature_data_dir)
        #
        # for data_file in data_file_list:
        #     # 时间太长，这里方便中断后继续进行
        #     if data_file in output_file_list:
        #         continue
        #     print("Processing data file:", data_file)
        #     feature_data = pd.DataFrame(columns=['feature_vector', 'label'])
        #     feature_vectors, labels = [], []
        #     file_index = data_file.strip().split('.')[0]
        #     file_path = os.path.join(newPrefix_data_dir, data_file)
        #     log_sequence_data = pd.read_csv(open(file_path, 'r+', encoding='utf-8'))
        #
        #     frequency_feature = frequency_extraction(log_sequence_data, self.log_templates)
        #     seasonality_feature = seasonality_extraction(log_sequence_data, frequency_feature,self.log_templates, self.time_bin)
        #     sequence_feature = sequence_extraction(log_sequence_data, lcs_seq)
        #     surge_feature = surge_extraction(log_sequence_data, self.log_templates, self.duration)
        #
        #     if surge_feature is None:
        #         continue
        #
        #     # print(frequency_feature.shape, seasonality_feature.shape, sequence_feature.shape, surge_feature.shape)
        #     features = np.concatenate((sequence_feature, frequency_feature * seasonality_feature,
        #                             surge_feature * seasonality_feature), axis=1)
        #     print(features.shape)
        #
        #     for feature in features:
        #         feature_vectors.append(' '.join(list(map(str, list(feature)))))
        #
        #     labels.extend(list(log_sequence_data['label']))
        #
        #     feature_data['feature_vector'] = feature_vectors
        #     feature_data['label'] = labels
        #
        #     feature_data.to_csv(feature_data_dir + file_index + '.csv', index=False)
