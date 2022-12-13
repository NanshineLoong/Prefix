from collections import Counter
import numpy as np

def frequency_extraction(dataset, log_templates):
    """
    计算频率特征
    """
    print('frequency extraction start')
    frequency_features = []
    for _, row in dataset.iterrows():
        frequency_features.append([0] * log_templates)
        log_sequence = row['template_ids'].strip().split()
        counts = Counter(log_sequence)
        for template_id, count in counts.items():
            frequency_features[-1][int(template_id)] = count

    print('frequency extraction end')

    return np.array(frequency_features)