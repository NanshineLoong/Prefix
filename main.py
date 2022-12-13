from exp.exp_random_forest import Exp_random_forest
import os
import sys
import warnings

if __name__ == '__main__':
    # 调试时进入当前目录
    if os.getcwd() != sys.path[0]:
        os.chdir(sys.path[0])
    
    warnings.filterwarnings("ignore")
    
    model = Exp_random_forest()
    model.train()  #当前设定仅对M1数据集做特征提取和训练
