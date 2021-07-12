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

dic_subject = {1: '15', 2: '16', 3: '19', 4: '20', 5: '21', 6: '22', 7: '23', 8: '24', 9: '25', 10: '26', 11: '27',
               12: '29', 13: '30', 14: '31', 15: '32', 16: '33', 17: '34', 18: '35', 19: '36', 20: '37', 21: '38',
               22: '40', }
dic_express = {'disgust1': '0101', 'disgust2': '0102', 'anger1': '0401', 'anger2': '0402', 'happy1': '0502',
               'happy2': '0503', 'happy3': '0505', 'happy4': '0507', 'happy5': '0508'}


def read_label_csv(label_csv):
    label_csv_path_list = os.path.split(label_csv)
    label_csv_dir = label_csv_path_list[0]
    label_csv_name = label_csv_path_list[1]
    label_csv_name_list = label_csv_name.split('.')
    label_csv_name_save = label_csv_name_list[0] + '_convert.' + label_csv_name_list[1]
    save_path = os.path.join(label_csv_dir, label_csv_name_save)
    if os.path.exists(save_path):
        df_label = pd.read_csv(save_path)
    else:
        df_label = pd.read_csv(label_csv)
        df_label['subject'] = df_label['subject'].map(dic_subject)
        df_label['express'] = df_label['express'].apply(lambda x:x.split('_')[0]).map(dic_express)
        df_label.to_csv(save_path, index=False)
    return df_label

def get_cols_names():
    global df_all_feats_cols_name
    columns_AU_r = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',
                    'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    columns_AU_c = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',
                    'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    df_std_cols_name = list(map(lambda column_name: column_name + '_std', columns_AU_r))
    # df_std_cols_name = [column_name + '_std' for column_name in columns_std_variation]
    df_variation_cols_name = list(map(lambda column_name: column_name + '_variation', columns_AU_r))
    AU_r_rolling_max_columns = list(map(lambda column: column + '_max', columns_AU_r))
    AU_c_rolling_frequency_columns = list(map(lambda column: column + '_frequency', columns_AU_c))
    df_all_feats_cols_name = df_std_cols_name + df_variation_cols_name + AU_r_rolling_max_columns + AU_c_rolling_frequency_columns


def extract_featrues_fixed_len(df, onset_offset_frame_list, openface_featrue_file_name, len_frame=180):
    onset_offset_frame_list_sorted = sorted(onset_offset_frame_list, key=(lambda x:x[0]))
    label_list = []
    for i, onset_offset_frame in enumerate(onset_offset_frame_list_sorted):
        onset_frame = onset_offset_frame[0]
        apex_frame = onset_offset_frame[1]
        offset_frame = onset_offset_frame[2]
        if offset_frame == 0:
            offset_frame = apex_frame + apex_frame - onset_frame
        if offset_frame-onset_frame>len_frame:
            print(f'file:{openface_featrue_file_name}, offset_frame:{offset_frame},onset_frame:{onset_frame} is over than {len_frame}!!!!')
        else:
            if i > 0:
                start = onset_offset_frame_list_sorted[i - 1][2]
            else:
                start = 0
            if i < len(onset_offset_frame_list_sorted) - 1:
                end = onset_offset_frame_list_sorted[i + 1][0]
            else:
                end = df.shape[0]
            if end-start<len_frame:
                print(f'file:{openface_featrue_file_name},start:{start},end:{end} is small than {len_frame}!!!!')
            else:
                distance_start = onset_frame - start
                distance_end = end - offset_frame
                offset = len_frame-(offset_frame-onset_frame)
                if distance_start < distance_end:
                    if distance_start > offset:
                        # start_crop = random.randint(onset_frame-offset, onset_frame)
                        start_crop = random.randint(offset_frame - len_frame, onset_frame)
                    else:
                        start_crop = random.randint(start, onset_frame)
                else:
                    if distance_end > offset:
                        # start_crop = random.randint(onset_frame - offset, onset_frame)
                        start_crop = random.randint(offset_frame - len_frame, onset_frame)
                    else:
                        start_crop = random.randint(start, end-len_frame)
                label_start = onset_frame - start_crop
                label_end = offset_frame - start_crop
                df_sampled = pd.DataFrame(data=df[start_crop:start_crop+len_frame], columns=df.columns)
                df_save_file = os.path.join(features_engineered_root, openface_featrue_file_name.split('.')[0]+'_'+str(i)+'.csv')
                df_sampled.to_csv(df_save_file, index=False)
                label_list.append([df_save_file, label_start, label_end])
        print(f'{openface_featrue_file_name} is done!')
    return label_list

def featrue_engineer_openface(openface_featrue_file_path, df_label_org, label_all):
    # get file names
    path_list = openface_featrue_file_path.split('/')
    openface_featrue_file_name = path_list[-1]
    # sub_dir = path_list[-2]
    # read a openface_featrue file
    # df_openface = pd.read_csv(openface_featrue_file_path)
    df_openface = pd.read_csv(
        openface_featrue_file_path,
        delimiter=",",
        engine='python',
        skipinitialspace=True
    ).fillna(0)
    df_openface = df_openface[(df_openface["success"] == 1)]
    df_openface_AU_r = df_openface.loc[:, 'AU01_r':'AU45_r']
    df_openface_AU_c = df_openface.loc[:, 'AU01_c':'AU45_c']

    # join into big DF, and seek for the range of std
    df_openface_std_variation_rolling = df_openface_AU_r.rolling(window=window_width)
    df_openface_rolling_std = df_openface_std_variation_rolling.std()
    df_openface_rolling_variation = df_openface_std_variation_rolling.max() - df_openface_std_variation_rolling.min()
    # seek for the maximum value of AU_r in rolling
    df_AU_r_rolling_max = df_openface_AU_r.rolling(window=window_width).max()
    # seek for the frequency of AU_c in rolling
    df_AU_c_rolling_frequency = df_openface_AU_c.rolling(window=window_width).sum() / window_width
    # combine all the features into one
    df_all_feats = pd.concat(
        [df_openface_rolling_std, df_openface_rolling_variation, df_AU_r_rolling_max, df_AU_c_rolling_frequency],
        ignore_index=True, axis=1).fillna(0)
    # give value of df_all_feats_cols_name to the columns of df_all_feats
    df_all_feats.columns = df_all_feats_cols_name
    df_save_dir = os.path.join(fme_dir, OpenFace_features_engineer_fold)
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir, exist_ok=True)
    # df_save_file = os.path.join(df_save_dir, openface_featrue_file_name)
    # extract features according to fixed frmae length of 200 and groundtruth duration of expression(onset frame and offset frame)
    file_name_prefix_list = openface_featrue_file_name[:7].split('_')
    subject = int(file_name_prefix_list[0])
    express = int(file_name_prefix_list[1])
    df_selected = df_label_org.loc[(df_label_org['subject'] == subject) & (df_label_org['express'] == express)]
    onset_offset_frame_list = df_selected[['onset_frame', 'apex_frame', 'offset_frame']].values
    label_list = extract_featrues_fixed_len(df_all_feats, onset_offset_frame_list, openface_featrue_file_name)
    label_all.extend(label_list)
    return label_all


def main():
    df_label_org = read_label_csv(csv_label_org)
    get_cols_names()
    # openface_features_csvs = [os.path.join(openface_features_root, 's25/25_0508funnydunkey.csv')]
    openface_features_csvs = glob(openface_features_root + '/*/*.csv')
    label_all = []
    # pool = multiprocessing.Pool(processes=processes_num)
    # for OpenFace_features_file in openface_features_csvs:
    #     pool.apply_async(func=featrue_engineer_openface, args=(OpenFace_features_file, df_label_org, label_all))
    # pool.close()
    # pool.join()
    for OpenFace_features_file in openface_features_csvs:
        label_all = featrue_engineer_openface(OpenFace_features_file, df_label_org, label_all)
    df_label = pd.DataFrame(data=label_all, columns=['file_path', 'frame_start', 'frame_end'])
    df_label.to_csv(os.path.join(features_engineered_root, 'label_croped.csv'), index=False)


if __name__ == "__main__":
    # parser config
    config_file = "../config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    fme_dir = cp["DEFAULT"].get("fme_dir")
    OpenFace_features_fold = cp["DEFAULT"].get("OpenFace_features_fold")
    csv_label_org = cp["DEFAULT"].get("csv_label_org")
    processes_num = cp["DEFAULT"].getint("processes_num")
    window_width = cp["DEFAULT"].getint("window_width")
    OpenFace_features_engineer_fold = f'features_engineered/window_{str(window_width)}'
    openface_features_root = os.path.join(fme_dir, OpenFace_features_fold)
    features_engineered_root = os.path.join(fme_dir, OpenFace_features_engineer_fold)
    main()
