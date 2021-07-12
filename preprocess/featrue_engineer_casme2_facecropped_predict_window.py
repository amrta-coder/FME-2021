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
        print(f'Load CSV:{save_path}')
        df_label = pd.read_csv(save_path)
    else:
        df_label = pd.read_csv(label_csv)
        df_label['subject'] = df_label['subject'].map(dic_subject)
        df_label['express'] = df_label['express'].apply(lambda x:x.split('_')[0]).map(dic_express)
        df_label.to_csv(save_path, index=False)
    return df_label


def extract_featrues_fixed_len(df, onset_offset_frame_list, openface_featrue_file_path, len_frame=8, step=4):
    print(f'{openface_featrue_file_path} is start!!!')
    label_list = []
    onset_offset_list = []
    onset_offset_frame_list_sorted = sorted(onset_offset_frame_list, key=(lambda x:x[0]))
    for onset_offset_frame in onset_offset_frame_list_sorted:
        onset_frame = onset_offset_frame[0]
        apex_frame = onset_offset_frame[1]
        offset_frame = onset_offset_frame[2]
        if offset_frame == 0:
            offset_frame = apex_frame + apex_frame - onset_frame
        onset_offset_list.extend(range(onset_frame, offset_frame+1))

    path_list = openface_featrue_file_path.split('/')
    openface_featrue_file_save = os.path.join(features_output_root, path_list[-2], path_list[-1].split('.')[0])
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
    mmsc = load(open('casme2_window_MinMaxScaler.pkl', 'rb'))
    df_openface = pd.DataFrame(mmsc.transform(df_openface), columns=df_openface.columns)
    df_save_dir = os.path.join(fme_dir_fc, OpenFace_features_engineer_fold)
    if not os.path.exists(df_save_dir):
        os.makedirs(df_save_dir, exist_ok=True)
    # extract features according to fix length of 8 and step of 4
    openface_featrue_file_name = openface_featrue_file_path.split('/')[-1]
    file_name_prefix_list = openface_featrue_file_name[:7].split('_')
    subject = int(file_name_prefix_list[0])
    express = int(file_name_prefix_list[1])
    df_selected = df_label_org.loc[(df_label_org['subject'] == subject) & (df_label_org['express'] == express)]
    onset_offset_frame_list = df_selected[['onset_frame', 'apex_frame', 'offset_frame']].values
    label_list = extract_featrues_fixed_len(df_openface, onset_offset_frame_list, openface_featrue_file_path)
    label_all.extend(label_list)
    return label_all

def sort_fun(path):
    file_list = path.str.split('/')
    file_names_int = int(file_list[-1].split('.')[0])
    return (file_list[-3], file_list[-2], file_names_int)

def main():
    df_label_org = read_label_csv(csv_label_org)
    label_all = []
    # openface_features_csvs = [os.path.join(openface_features_root, 's15/15_0101disgustingteeth.csv')]
    openface_features_csvs = glob(features_input_root + '/*/*.csv')
    for OpenFace_features_file in openface_features_csvs:
        label_all = featrue_engineer_openface(OpenFace_features_file, df_label_org, label_all)
    df_label = pd.DataFrame(data=label_all, columns=['file_path', 'label'])
    df_label['object'] = df_label['file_path'].apply(lambda x: x.split('/')[-3])
    df_label['subdir'] = df_label['file_path'].apply(lambda x: x.split('/')[-2])
    df_label['num'] = df_label['file_path'].apply(lambda x: x.split('/')[-1].split('.')[0]).astype(int)
    df_label.sort_values(by=['object', 'subdir', 'num'], inplace=True)
    df_label[['file_path', 'label']].to_csv(os.path.join(features_output_root, label_casme2_predict_window), index=False)


if __name__ == "__main__":
    # parser config
    config_file = "../config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    fme_dir_fc = cp["DEFAULT"].get("fme_dir_fc")
    csv_label_org = cp["DEFAULT"].get("csv_label_org_fc")
    processes_num = cp["DEFAULT"].getint("processes_num")
    label_casme2_predict_window = cp["TEST"].get("label_casme2_predict_window")
    OpenFace_features_engineer_fold = f'features_engineered/predictions_windows'
    features_engineer_AU_windows_fold = f'features_engineered/AU_windows'
    features_input_root = os.path.join(fme_dir_fc, features_engineer_AU_windows_fold)
    features_output_root = os.path.join(fme_dir_fc, OpenFace_features_engineer_fold)
    main()
