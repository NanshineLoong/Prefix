import numpy as np
import pandas as pd
from fastsst import SingularSpectrumTransformation
from tqdm import tqdm

def generate_time_sequence(template_sequence, log_templates, duration):
    """
    将模板序列转化为每个模板数量变化的时间序列
    """
    time_sequence = []
    start_time, end_time = int(template_sequence['start_time']), int(template_sequence['end_time'])
    template_ids = template_sequence['template_ids'].strip().split()
    timestamps = template_sequence['timestamps'].strip().split()
    period_end_time = start_time + duration
    index = 0
    while period_end_time <= end_time:
        template_num = [0] * log_templates
        while index < len(timestamps) and int(timestamps[index]) <= period_end_time:
            template_num[int(template_ids[index])] += 1
            index += 1
        time_sequence.append(template_num)
        period_end_time += duration

    time_sequence = np.array(time_sequence).T
    return time_sequence

def surge_extraction(dataset, log_templates,duration):
    """
    计算突变特征
    """
    print('surge extraction start')
    features = np.array(pd.DataFrame(columns=range(log_templates)))
    for _, row in tqdm(dataset.iterrows()):
        time_sequences = generate_time_sequence(row, log_templates, duration)
        surge_feature = np.array(pd.DataFrame(columns=range(60)))
        for seq in time_sequences:
            try:
                change_score = SingularSpectrumTransformation(win_length=15).score_offline(seq)
            except Exception:
                try:
                    change_score = SingularSpectrumTransformation(win_length=20).score_offline(seq)
                except Exception:
                    return None
            surge_feature = np.append(surge_feature, change_score.reshape(1, -1), axis=0)

        surge_feature = np.max(surge_feature, axis=1).reshape((1, -1))
        features = np.append(features, surge_feature, axis=0)

    features = features.reshape(-1, log_templates)
    print('surge extraction end')
    return features