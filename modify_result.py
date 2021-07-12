import numpy as np
import os
from configparser import ConfigParser
from generator_vad import FeatruesSequence
import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_optimal_precision_recall, pred_modify


def main():
    test_csv = os.path.join('/home/zzg/workspace/pycharm/FME_2021/experiments/20210612-172132_casme2_GRU/result.csv')
    df_test = pd.read_csv(test_csv)
    df_test.sort_values()
    prob_array = df_test['probs']
    pred_threshold = df_test['pred_threshold']
    df_test['pred_threshold_modify'] = pred_modify(pred_threshold)
    for thresh in np.linspace(start=0.1, stop=0.9, num=9):
        pred_threshold = (prob_array > thresh).astype(int)
        df_test[f'pred_threshold_{str(thresh)}'] = pred_threshold
        df_test[f'pred_threshold_{str(thresh)}_modify'] = pred_modify(pred_threshold)
    df_test.to_csv(test_csv, index=False)

if __name__ == "__main__":
    main()
