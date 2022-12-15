from tqdm import tqdm
import numpy as np


def frequency_extraction(dataset, log_templates):
    templates = []
    time_bins = []
    for _, row in dataset.iterrows():
        line = row['template_ids'].strip().split()
        templates.append(line)
        time_bins.append(row['timestamps'])

    t_l = len(time_bins)
    # 创建一个time_bins * templates的0矩阵
    data = np.zeros((t_l, log_templates)).astype(int)

    for i in tqdm(range(t_l), desc="frequency extraction:"):
        for t in templates[i]:
            t = int(t)
            data[i][t] = data[i][t] + 1

    return np.array(data)
