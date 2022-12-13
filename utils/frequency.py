from collections import Counter
import numpy as np


def frequency_extraction(dataset, log_templates):
    templates = []
    timebins = []
    for _, row in dataset.iterrows():
        line = row['template_ids'].strip().split()
        templates.append(line)
        timebins.append(row['timestamps'])

    t_l = len(timebins)
    # 创建一个timebins * templates的0矩阵
    data = np.zeros((t_l, log_templates)).astype(int)

    for i in range(t_l):
        for t in templates[i]:
            t = int(t)
            data[i][t] = data[i][t] + 1

    return np.array(data)