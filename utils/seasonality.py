import numpy as np
from tqdm import *
import pandas as pd


def cal_period_difference(dataset, frequency_features, period_time, log_templates, time_bin):
    """
    计算给定周期内模板数量差的平方和
    """
    start_index, end_index = 0, 0
    period_difference = np.array([0] * log_templates, dtype='int64')
    frequency_features = np.array(frequency_features)
    start_time = int(dataset['end_time'][start_index])

    while end_index < len(dataset):
        end_time = int(dataset['end_time'][end_index])
        periods = (end_time - start_time) // time_bin
        if periods < period_time:
            end_index += 1
        elif periods == period_time:
            period_difference += np.square(frequency_features[end_index] - frequency_features[start_index])
            start_index += 1
            end_index += 1
            start_time = int(dataset['end_time'][start_index])
        else:
            period_difference += np.square(frequency_features[start_index])
            start_index += 1
            start_time = int(dataset['end_time'][start_index])

    return period_difference


def seasonality_extraction(dataset, frequency_features, log_templates, time_bin):
    """
    计算周期特征
    """
    # print('seasonality extraction start')
    candidate_periods = [4, 96, 672, 2880]
    differences = []
    for i in range(candidate_periods[-1]):
        period_difference = cal_period_difference(dataset, frequency_features, i + 1, log_templates, time_bin)
        if len(differences) == 0:
            differences.append(period_difference)
        else:
            differences.append(period_difference + differences[-1])

    candidate_difference = np.array(pd.DataFrame(columns=range(log_templates)))
    for p in candidate_periods:
        diff_avg = differences[p - 1] / p
        cur_diff = differences[p - 1].astype(np.float64) - differences[p - 2].astype(np.float64)
        candidate_difference = \
            np.append(candidate_difference, np.divide(cur_diff, diff_avg, out=np.ones_like(cur_diff)*np.inf,
                                                        where=diff_avg != 0).reshape((1, -1)), axis=0)

    seasonality_feature = np.min(candidate_difference, axis=0)

    # print('seasonality extraction end')

    return seasonality_feature.reshape(1, -1)