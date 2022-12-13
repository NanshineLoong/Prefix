from tqdm import *
import Levenshtein
import pickle
import heapq
from random import sample
import numpy as np


def lcs(seq1, seq2):
    lengths = [[0 for _ in range(len(seq2) + 1)] for _ in range(len(seq1) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    # read the substring out from the matrix
    result = []
    lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
    while lenOfSeq1 != 0 and lenOfSeq2 != 0:
        if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 - 1][lenOfSeq2]:
            lenOfSeq1 -= 1
        elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2 - 1]:
            lenOfSeq2 -= 1
        else:
            assert seq1[lenOfSeq1 - 1] == seq2[lenOfSeq2 - 1]
            result.insert(0, seq1[lenOfSeq1 - 1])
            lenOfSeq1 -= 1
            lenOfSeq2 -= 1
    return result


# 这个函数用来计算平均相关性（平均得分）
# 就是计算那100条omen和1w条nonomen序列每一条的平均得分
def avg_scores(seq, sequences):
    score = list()
    for seq_2 in sequences:
        currentlcssequence = lcs(seq, seq_2)  # 这里是得到lcs set的函数，在最上面
        score.append(len(currentlcssequence)/len(seq_2))  # 把得分保存 等会计算平均值
    return np.mean(score)


# 这个函数是计算100条中每个序列的得分，并将得分排序选取前三名
# 输入的第一个参数是lcs set, 第二个参数是选取的100条Omen序列, 第三个是选区的1w条nonomen序列, 第四个是选取的序列的数量
def select_lcs_sequence(lcs_list, sample_omen_sequence, sample_non_omen_sequence, size=2):
    seq_score = dict()
    final_lcs_set = set()
    for seq in tqdm(lcs_list):
        score = avg_scores(seq, sample_omen_sequence) - avg_scores(seq, sample_non_omen_sequence)
        seq_score[' '.join(seq)] = score

    final_lcs_list = heapq.nlargest(size, seq_score)
    final_lcs_set = set(final_lcs_list)

    return final_lcs_set
    # best_lcs_set1 = []
    # best_lcs_set2 = []
    # best_score1 = 0  # 得分最高的
    # best_score2 = 0.00001# 得分第二高的
    # for i in range(len(new_lcs_set)):
    #     # 遍历的长度就是100 因为我们选了100个lcs set中的序列
    #     score = avg_scores(new_lcs_set[i],lcs_set) - avg_scores(new_lcs_set[i],nonomen_sequence)
    #     if score > best_score2:
    #         best_score2 = score
    #         best_lcs_set2 = lcs_set[i]
    #     if best_score2 > best_score1:
    #         best_score1,best_score2 = best_score2,best_score1  # 交换两个值
    #         best_lcs_set1,best_lcs_set2 = best_lcs_set2,best_lcs_set1
    # return best_lcs_set1,best_lcs_set2


def get_log_lists(train_data, mode):
    sequence = train_data

    if mode == 'omen':
        sequence = train_data[train_data['label'] == 1]
    elif mode == 'non-omen':
        sequence = train_data[train_data['label'] == 0]
    sequence = sequence.reset_index(drop=True)
    log_sequence = set()
    for index, row in sequence.iterrows():
        # if index == 0 or int(row['end_time']) - int(omen_sequence['end_time'][index - 1]) > 7200:
        log_sequence.add(row['template_ids'])

    log_lists = list()
    for seq in log_sequence:
        log_lists.append(seq.strip().split())

    return log_lists

# 获取所有预兆序列的lcs
def get_lcs_set(train_data, output_dir, size):
    lcs_set = set()

    print("generating LCS set...")
    omen_logs = get_log_lists(train_data, "omen")
    for i in tqdm(range(1, len(omen_logs))):
        for j in range(i - 1, -1, -1):
            lcs_set.add(' '.join(lcs(omen_logs[i], omen_logs[j])))

    lcs_list = []
    for seq in lcs_set:
        lcs_list.append(seq.strip().split())

    # 根据长度过滤
    def control(x):
        return 4 <= len(x) <= 20

    lcs_list = list(filter(control, list(lcs_list)))
    sample_omen_sequence = sample(omen_logs, 100)

    non_omen_logs = get_log_lists(train_data, "non-omen")
    sample_non_omen_sequence = sample(non_omen_logs, 10000)

    final_lcs_set = select_lcs_sequence(lcs_list, sample_omen_sequence, sample_non_omen_sequence, size)

    pickle.dump(final_lcs_set, open(output_dir, 'wb+'))

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