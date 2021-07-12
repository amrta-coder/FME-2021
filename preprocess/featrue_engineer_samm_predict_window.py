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
from pickle import dump, load


def extract_featrues_fixed_len(df, onset_offset_frame_list, openface_featrue_file_path, len_frame=40, step=20):
    print(f'{openface_featrue_file_path} is start!!!')
    label_list = []
    onset_offset_list = []
    onset_offset_frame_list_sorted = sorted(onset_offset_frame_list, key=(lambda x:x[0]))
    for onset_offset_frame in onset_offset_frame_list_sorted:
        onset_offset_list.extend(range(onset_offset_frame[0], onset_offset_frame[1]+1))

    openface_featrue_file_save = os.path.join(features_output_root, openface_featrue_file_path.split('.')[0])
    os.makedirs(openface_featrue_file_save, exist_ok=True)
    end = df.shape[0]
    count =0
    for i in range(0, end, step):
        label = 0
        offset = i + len_frame
        if offset < end:
            df_sampled = df[i:offset]
            df_save_file = os.path.join(openface_featrue_file_save, str(count) + '.csv')
            df_sampled.to_csv(df_save_file, index=False)
            in_label_list = np.intersect1d(list(range(i, offset)), onset_offset_list)
            if len(in_label_list) > len_frame/2:
                label = 1
            label_list.append([df_save_file, label])
            count = count + 1
    print(f'{openface_featrue_file_path} is done')
    return label_list

def featrue_engineer_openface(openface_featrue_file_path, df_label_org, label_all):
    df_openface = pd.read_csv(
        openface_featrue_file_path,
        delimiter=",",
        engine='python',
        skipinitialspace=True
    ).fillna(0)
    mmsc = load(open('samm_window_MinMaxScaler.pkl', 'rb'))
    df_openface = pd.DataFrame(mmsc.transform(df_openface), columns=df_openface.columns)
    df_save_dir = features_output_root
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir, exist_ok=True)
    # extract features according to fix length of 40 and step of 20
    openface_featrue_file_name = openface_featrue_file_path.split('/')[-1]
    csv_name = openface_featrue_file_name.split('.')[0]
    df_selected = df_label_org.loc[(df_label_org['Filename'].str[:5] == csv_name)]
    onset_offset_frame_list = df_selected[['Onset', 'Offset']].values
    label_list = extract_featrues_fixed_len(df_openface, onset_offset_frame_list, openface_featrue_file_name)
    label_all.extend(label_list)
    return label_all

def sort_fun(path):
    file_list = path.str.split('/')
    file_names_int = int(file_list[-1].split('.')[0])
    return (file_list[-3], file_list[-2], file_names_int)

def main():
    df_label_org = pd.read_csv(csv_label_org_samm)
    label_all = []
    # openface_features_csvs = [os.path.join(openface_features_root, 's15/15_0101disgustingteeth.csv')]
    openface_features_csvs = glob(features_input_root + '/*.csv')
    for OpenFace_features_file in openface_features_csvs:
        label_all = featrue_engineer_openface(OpenFace_features_file, df_label_org, label_all)
    df_label = pd.DataFrame(data=label_all, columns=['file_path', 'label'])
    df_label['subdir'] = df_label['file_path'].apply(lambda x: x.split('/')[-2])
    df_label['num'] = df_label['file_path'].apply(lambda x: x.split('/')[-1].split('.')[0]).astype(int)
    df_label.sort_values(by=['subdir', 'num'], inplace=True)
    df_label[['file_path', 'label']].to_csv(os.path.join(features_output_root, label_samm_predict_window), index=False)


if __name__ == "__main__":
    # parser config
    config_file = "../config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    fme_dir_samm = cp["DEFAULT"].get("fme_dir_samm")
    OpenFace_features_fold_samm = cp["DEFAULT"].get("OpenFace_features_fold_samm")
    csv_label_org_samm = cp["DEFAULT"].get("csv_label_org_samm")
    processes_num = cp["DEFAULT"].getint("processes_num")
    label_samm_predict_window = cp["TEST"].get("label_samm_predict_window")
    OpenFace_features_engineer_fold = f'features_engineered/predictions_windows'
    features_engineer_AU_windows_fold = f'features_engineered/AU_windows'
    features_input_root = os.path.join(fme_dir_samm, features_engineer_AU_windows_fold)
    features_output_root = os.path.join(fme_dir_samm, OpenFace_features_engineer_fold)
    csv_label_org_samm = os.path.join(fme_dir_samm, csv_label_org_samm)
    main()
