import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser
import multiprocessing
from math import ceil
from glob import glob
from easydict import EasyDict as edict
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pickle import dump, load


def create_tranformer(type):
    AU_windows_features_dir = eval('AU_windows_features_dir_'+type)
    AU_windows_features_file_list = glob(os.path.join(AU_windows_features_dir, '*.csv'))
    df_list = []
    for openface_featrue_file_path in AU_windows_features_file_list:
        df_AU_windows_features = pd.read_csv(
                openface_featrue_file_path,
                delimiter=",",
                engine='python',
                skipinitialspace=True
            ).fillna(0)
        df_list.append(df_AU_windows_features)
    df_AU_windows_features_all = pd.concat(df_list)

    stdsc = StandardScaler()
    mmsc = MinMaxScaler()
    df_openface_AU_r = mmsc.fit_transform(df_AU_windows_features_all)
    dump(mmsc, open(f'{type}_window_MinMaxScaler.pkl', 'wb'))

    df_openface_AU_r = stdsc.fit_transform(df_AU_windows_features_all)
    dump(stdsc, open(f'{type}_window_StandardScaler.pkl', 'wb'))

if __name__ == '__main__':
    AU_windows_features_dir_casme2 = '/home/zzg/data/micro_expression/CASME^2_longVideoFaceCropped/features_engineered/AU_windows/*'
    AU_windows_features_dir_samm = '/home/zzg/data/micro_expression/SAMM/features_engineered/AU_windows'
    for type in ['casme2', 'samm']:
        create_tranformer(type)