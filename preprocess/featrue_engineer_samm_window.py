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


def extract_featrues_fixed_len(df, onset_offset_frame_list, openface_featrue_file_name, len_frame=40, step=20):
    print(f'{openface_featrue_file_name} is start!!!')
    label_list = []
    end = df.shape[0]
    count_pos = 0
    count_neg = 0
    onset_offset_frame_list_sorted = sorted(onset_offset_frame_list, key=(lambda x:x[0]))

    for onset_offset_frame in onset_offset_frame_list_sorted:
        print(f'{onset_offset_frame} is start!!!')
        onset_frame = onset_offset_frame[0]
        offset_frame = onset_offset_frame[1]
        if offset_frame - onset_frame < len_frame and offset_frame - onset_frame > len_frame / 2:
            df_sampled = df[onset_frame:onset_frame + len_frame]
            df_save_file = os.path.join(features_output_root,
                                        openface_featrue_file_name.split('.')[0] + '_pos_' + str(onset_frame) + '.csv')
            df_sampled.to_csv(df_save_file, index=False)
            label_list.append([df_save_file, 1])
            count_pos += 1
        else:
            for i in range(onset_frame, offset_frame, step):
                if i + len_frame < min(offset_frame, end):
                    df_sampled = df[i:i+len_frame]
                    df_save_file = os.path.join(features_output_root,
                                                openface_featrue_file_name.split('.')[0] + '_pos_' + str(i) + '.csv')
                    df_sampled.to_csv(df_save_file, index=False)
                    label_list.append([df_save_file, 1])
                    count_pos += 1
        print(f'{openface_featrue_file_name} crop the pos number is {count_pos}')
    # df_neg_all = pd.concat(df_neg_list)
    drop_list = []
    for onset_offset in onset_offset_frame_list_sorted:
        drop_list.extend(range(onset_offset[0], onset_offset[1]))
    df_neg_all = df.drop(drop_list)
    end_neg = df_neg_all.shape[0]
    for i in range(0, end_neg, len_frame):
        print(f'crop negative csv is start!!!')
        if i + len_frame < end_neg and count_neg < count_pos * 1.5:
            df_sampled = df[i:i+len_frame]
            df_save_file = os.path.join(features_output_root,
                                        openface_featrue_file_name.split('.')[0] + '_neg_' + str(i) + '.csv')
            df_sampled.to_csv(df_save_file, index=False)
            label_list.append([df_save_file, 0])
            count_neg += 1
    print(f'{openface_featrue_file_name} is done!')
    return label_list

def featrue_engineer_openface(openface_featrue_file_path, df_label_org, label_all):
    # get file names
    openface_featrue_file_name = openface_featrue_file_path.split('/')[-1]
    # sub_dir = path_list[-2]
    # read a openface_featrue file
    # df_openface = pd.read_csv(openface_featrue_file_path)
    df_openface = pd.read_csv(
        openface_featrue_file_path,
        delimiter=",",
        engine='python',
        skipinitialspace=True
    ).fillna(0)
    mmsc = load(open('samm_window_MinMaxScaler.pkl', 'rb'))
    df_openface_AU_r = pd.DataFrame(mmsc.transform(df_openface), columns=df_openface.columns)
    df_save_dir = os.path.join(fme_dir_samm, OpenFace_features_engineer_fold)
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir, exist_ok=True)
    # extract features according to fix length of 40 and step of 20
    csv_name = openface_featrue_file_name.split('.')[0]
    df_selected = df_label_org.loc[(df_label_org['Filename'].str[:5] == csv_name)]
    onset_offset_frame_list = df_selected[['Onset', 'Offset']].values
    label_list = extract_featrues_fixed_len(df_openface_AU_r, onset_offset_frame_list, openface_featrue_file_name)
    label_all.extend(label_list)
    return label_all


def main():
    df_label_org = pd.read_csv(csv_label_org_samm)
    # openface_features_csvs = [os.path.join(openface_features_root, 's23/23_0102eatingworms.csv')]
    openface_features_csvs = glob(features_input_root + '/*.csv')
    label_all = []
    for OpenFace_features_file in openface_features_csvs:
        label_all = featrue_engineer_openface(OpenFace_features_file, df_label_org, label_all)
    df_label = pd.DataFrame(data=label_all, columns=['file_path', 'label'])
    df_label.to_csv(os.path.join(features_output_root, label_samm_window), index=False)


if __name__ == "__main__":
    # parser config
    config_file = "../config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    fme_dir_samm = cp["DEFAULT"].get("fme_dir_samm")
    OpenFace_features_fold_samm = cp["DEFAULT"].get("OpenFace_features_fold_samm")
    csv_label_org_samm = cp["DEFAULT"].get("csv_label_org_samm")
    label_samm_window = cp["DEFAULT"].get("label_samm_window")
    processes_num = cp["DEFAULT"].getint("processes_num")
    OpenFace_features_engineer_fold = f'features_engineered/VAD_window'
    features_engineer_AU_windows_fold = f'features_engineered/AU_windows'
    features_input_root = os.path.join(fme_dir_samm, features_engineer_AU_windows_fold)
    features_output_root = os.path.join(fme_dir_samm, OpenFace_features_engineer_fold)
    csv_label_org_samm = os.path.join(fme_dir_samm, csv_label_org_samm)
    main()
