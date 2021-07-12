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
from tqdm import tqdm


dic_subject = {1: '15', 2: '16', 3: '19', 4: '20', 5: '21', 6: '22', 7: '23', 8: '24', 9: '25', 10: '26', 11: '27',
               12: '29', 13: '30', 14: '31', 15: '32', 16: '33', 17: '34', 18: '35', 19: '36', 20: '37', 21: '38',
               22: '40', }
dic_express = {'disgust1': '0101', 'disgust2': '0102', 'anger1': '0401', 'anger2': '0402', 'happy1': '0502',
               'happy2': '0503', 'happy3': '0505', 'happy4': '0507', 'happy5': '0508'}

def get_cols_names():
    global df_all_feats_cols_name
    columns_AU_r = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',
                    'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    df_std_cols_name = list(map(lambda column_name: column_name + '_std', columns_AU_r))
    df_variation_cols_name = list(map(lambda column_name: column_name + '_variation', columns_AU_r))
    AU_r_rolling_max_columns = list(map(lambda column: column + '_max', columns_AU_r))
    df_all_feats_cols_name = df_std_cols_name + df_variation_cols_name + AU_r_rolling_max_columns

def read_label_csv(label_csv):
    label_csv_path_list = os.path.split(label_csv)
    label_csv_dir = label_csv_path_list[0]
    label_csv_name = label_csv_path_list[1]
    label_csv_name_list = label_csv_name.split('.')
    label_csv_name_save = label_csv_name_list[0] + '_convert.' + label_csv_name_list[1]
    save_path = os.path.join(label_csv_dir, label_csv_name_save)
    if os.path.exists(save_path):
        print(f'Load CSV:{save_path}')
        df_label = pd.read_csv(save_path)
    else:
        df_label = pd.read_csv(label_csv)
        df_label['subject'] = df_label['subject'].map(dic_subject)
        df_label['express'] = df_label['express'].apply(lambda x:x.split('_')[0]).map(dic_express)
        df_label.to_csv(save_path, index=False)
    return df_label


def extract_featrues_fixed_len(df, onset_offset_frame_list, openface_featrue_file_name, len_frame=8, step=4):
    print(f'{openface_featrue_file_name} is start!!!')
    label_list = []
    end = df.shape[0]
    count_pos = 0
    count_neg = 0
    df_neg_list = []
    onset_offset_list = []
    onset_offset_frame_list_sorted = sorted(onset_offset_frame_list, key=(lambda x:x[0]))

    for i, onset_offset_frame in enumerate(onset_offset_frame_list_sorted):
        print(f'{onset_offset_frame} is start!!!')
        onset_frame = onset_offset_frame[0]
        apex_frame = onset_offset_frame[1]
        offset_frame = onset_offset_frame[2]
        if offset_frame == 0:
            offset_frame = apex_frame + apex_frame - onset_frame
        onset_offset_list.append([onset_frame, offset_frame])
        for i in range(onset_frame, offset_frame, step):
            if i + len_frame < min(offset_frame, end):
                df_sampled = df[i:i+len_frame]
                df_save_file = os.path.join(features_engineered_root,
                                            openface_featrue_file_name.split('.')[0] + '_pos_' + str(i) + '.csv')
                df_sampled.to_csv(df_save_file, index=False)
                label_list.append([df_save_file, 1])
                count_pos += 1
        print(f'{openface_featrue_file_name} crop the pos number is {count_pos}')
    # df_neg_all = pd.concat(df_neg_list)
    drop_list = []
    for onset_offset in onset_offset_list:
        drop_list.extend(range(onset_offset[0], onset_offset[1]+1))
    df_neg_all = df.drop(drop_list)
    end_neg = df_neg_all.shape[0]
    for i in range(0, end_neg, len_frame):
        print(f'crop negative csv is start!!!')
        if i + len_frame < end_neg and count_neg < count_pos * 1.5:
            df_sampled = df[i:i+len_frame]
            df_save_file = os.path.join(features_engineered_root,
                                        openface_featrue_file_name.split('.')[0] + '_neg_' + str(i) + '.csv')
            df_sampled.to_csv(df_save_file, index=False)
            label_list.append([df_save_file, 0])
            count_neg += 1
    print(f'{openface_featrue_file_name} is done!')
    return label_list

def featrue_engineer_openface(openface_featrue_file_path):
    # get file names
    path_list = openface_featrue_file_path.split('/')
    df_openface = pd.read_csv(
        openface_featrue_file_path,
        delimiter=",",
        engine='python',
        skipinitialspace=True
    ).fillna(0)
    df_openface.set_index('frame', inplace=True)
    df_openface_AU_r = df_openface.loc[:, 'AU01_r':'AU45_r']

    df_save_dir = os.path.join(fme_dir_fc, OpenFace_features_engineer_fold, path_list[-2])
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir, exist_ok=True)
    df_openface_AU_r.to_csv(os.path.join(df_save_dir, path_list[-1]), index=False)

    # join to big DF, and calculate the range of std
    df_openface_std_variation_rolling = df_openface_AU_r.rolling(window=window_width)
    df_openface_rolling_std = df_openface_std_variation_rolling.std()
    df_openface_rolling_variation = df_openface_std_variation_rolling.max() - df_openface_std_variation_rolling.min()
    # calculate the maximum of AU_r in rolling
    df_AU_r_rolling_max = df_openface_AU_r.rolling(window=window_width).max()
    df_all_feats = pd.concat([df_openface_rolling_std, df_openface_rolling_variation, df_AU_r_rolling_max],
        ignore_index=True, axis=1).fillna(0)
    # send the value of df_all_feats_cols_name to columns of df_all_feats
    df_all_feats.columns = df_all_feats_cols_name

    df_save_dir_window = os.path.join(fme_dir_fc, OpenFace_features_engineer_fold_window, path_list[-2])
    if not os.path.exists(df_save_dir_window):
        os.makedirs(df_save_dir_window, exist_ok=True)
    df_all_feats.to_csv(os.path.join(df_save_dir_window, path_list[-1]), index=False)


def main():
    get_cols_names()
    # openface_features_csvs = [os.path.join(openface_features_root, 's23/23_0102eatingworms.csv')]
    openface_features_csvs = glob(openface_features_root + '/*/*.csv')
    for OpenFace_features_file in tqdm(openface_features_csvs):
        featrue_engineer_openface(OpenFace_features_file)


if __name__ == "__main__":
    # parser config
    config_file = "../config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    fme_dir_fc = cp["DEFAULT"].get("fme_dir_fc")
    OpenFace_features_fold = cp["DEFAULT"].get("OpenFace_features_fold_fc")
    processes_num = cp["DEFAULT"].getint("processes_num")
    window_width = cp["DEFAULT"].getint("window_width")
    OpenFace_features_engineer_fold = f'features_engineered/AU'
    OpenFace_features_engineer_fold_window = f'features_engineered/AU_windows'
    openface_features_root = os.path.join(fme_dir_fc, OpenFace_features_fold)
    features_engineered_root = os.path.join(fme_dir_fc, OpenFace_features_engineer_fold)
    main()
