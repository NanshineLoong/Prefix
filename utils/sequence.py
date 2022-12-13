from tqdm import *
import Levenshtein

def sequence_extraction(dataset, lcs_set):
    """
    计算LCS2对应的相似度（序列特征）
    """
    print('sequence extraction start')
    lcs_list = []
    for s in lcs_set:
        lcs_list.append(s.strip().split())

    def cal_seq_sim(x):
        sequence_similarity = 0
        for seq in lcs_list:
            # lcs_length = len(lcs(x, seq))
            # sim = lcs_length / len(seq) if len(seq) > 0 else 0
            sim = Levenshtein.seqratio(x, seq)
            sequence_similarity = max(sequence_similarity, sim)
        return sequence_similarity

    template_sequence = dataset['template_ids'].apply(lambda x: x.strip().split())

    tqdm.pandas(desc='apply')  # 添加进度条
    sequence_sim = template_sequence.progress_apply(cal_seq_sim)

    print('sequence extraction end')

    return sequence_sim.values.reshape(-1, 1)