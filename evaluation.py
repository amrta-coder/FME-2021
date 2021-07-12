import os
import pandas as pd


def process_label_file(label_file_path):

    dic_subject = {1: '15', 2: '16', 3: '19', 4: '20', 5: '21', 6: '22', 7: '23', 8: '24',
                   9: '25', 10: '26', 11: '27', 12: '29', 13: '30', 14: '31', 15: '32', 16: '33',
                   17: '34', 18: '35', 19: '36', 20: '37', 21: '38', 22: '40'}
    dic_express = {'disgust1': '0101', 'disgust2': '0102', 'anger1': '0401', 'anger2': '0402', 'happy1': '0502',
                   'happy2': '0503', 'happy3': '0505', 'happy4': '0507', 'happy5': '0508'}
    df_label = pd.DataFrame(
        columns=['Video_ID', 'GT_onset', 'GT_offset', 'Duration'])

    if "CAS(ME)^2code_final" in label_file_path:
        df_label_original = pd.read_excel(label_file_path, header=None, engine='openpyxl')

        for index, row in df_label_original.iterrows():
            video_id = dic_subject[row[0]] + '_' + dic_express[row[1].split('_')[0]]
            onset = row[2]
            apex = row[3]
            offset = row[4]
            if offset == 0:
                offset = apex + (apex - onset)
            duration = offset - row[2]
            data = dict({'Video_ID': video_id, 'GT_onset': onset, 'GT_offset': offset, 'Duration': duration})
            df_label = df_label.append(data, ignore_index=True)

    if "SAMM" in label_file_path:
        df_label_original = pd.read_excel(label_file_path, header=9, engine='openpyxl')

        for index, row in df_label_original.iterrows():
            video_id = row['Filename'].split('_')[0] + '_' + row['Filename'].split('_')[1]
            data = dict({'Video_ID': video_id, 'GT_onset': row['Onset'],
                         'GT_offset': row['Offset'], 'Duration': row['Duration']})
            df_label = df_label.append(data, ignore_index=True)

    return df_label


def process_result_file(df_pred, frame_len, frame_step, strategy_num,
                        data_type='CASME2', label_type='pred_threshold_modify'):

    # strategy： if the number of 0 after 1 is lower than strategy_num, then change 0 to 1
    pre_label = 0
    pre_video = 0
    onset_seg = 0
    offset_seg = 0
    onset = 0
    offset = 0
    count = 0

    df_result = pd.DataFrame(
        columns=['Video_ID', 'Predicted_onset', 'Predicted_offset', 'Duration'])

    for index, row in df_pred.iterrows():
        if data_type == "CASME2":
            video_id = row['file_path'].split('/')[9][0:7]
            segment_num = row['file_path'].split('/')[10].split('.')[0]
        elif data_type == "SAMM":
            video_id = row['file_path'].split('/')[8]
            segment_num = row['file_path'].split('/')[9].split('.')[0]
        else:
            raise Exception("data_type should be either CASME2 or SAMM")

        if video_id == pre_video:  # the same VideoID
            if pre_label == 0:
                if row[label_type] == 1:  # just changed from 0 to 1
                    onset_seg = int(segment_num)
                    offset_seg = int(segment_num)
                    pre_label = 1
            elif pre_label == 1:
                if row[label_type] == 0:
                    if count < strategy_num:        # change types:   1 -> 0,  10 -> 0, 100 -> 0
                        count = count + 1
                    else:                           # changed type:   1000 -> 0
                        offset_seg = int(segment_num) - strategy_num - 1
                        onset = onset_seg * frame_step + 1
                        offset = offset_seg * frame_step + frame_len
                        duration = offset - onset
                        data = dict({'Video_ID': video_id, 'Predicted_onset': onset, 'Predicted_offset': offset,
                                     'Duration': duration})
                        df_result = df_result.append(data, ignore_index=True)

                        # initiate all the counter
                        pre_label = 0
                        onset_seg = 0
                        offset_seg = 0
                        onset = 0
                        offset = 0
                        count = 0
                else:       # change types: 1 -> 1,  10 -> 1, 100 -> 1, 1000 -> 1
                    pre_label = 1
                    offset_seg = int(segment_num)
                    count = 0
        else:     # change to a new video id
            # initiate all the counter
            pre_video = video_id
            onset_seg = 0
            offset_seg = 0
            onset = 0
            offset = 0
            pre_label = 0
            count = 0

            if row[label_type] == 1:
                pre_label = 1
                onset_seg = int(segment_num)

    return df_result


def process_result_file_best_strategy(df_pred, frame_len, frame_step, strategy_folder,
                                      data_type="CASME2", label_type="pred_threshold_modify"):

    # calculate the best strategy for each subject
    def find_subject_best_strategy(strategy_folder):
        list = {}
        for path, dir_list, file_list in os.walk(strategy_folder):
            for file_name in sorted(file_list):
                if not file_name.startswith('.') and (path.find("/.") == -1):
                    csv_file = os.path.join(path, file_name)
                    df = pd.read_csv(csv_file)
                    sorted_df = df.sort_values('F1_Score', ascending=False)
                    best_f1_strategy = sorted_df.iloc[0, 0]
                    subject = file_name.split('-')[0]
                    list[subject] = best_f1_strategy
        return list

    strategy_list = find_subject_best_strategy(strategy_folder)
    print(strategy_list)

    # strategy： if the number of 0 after 1 is lower than strategy_num, then change 0 to 1
    pre_label = 0
    pre_video = 0
    onset_seg = 0
    offset_seg = 0
    onset = 0
    offset = 0
    count = 0

    df_result = pd.DataFrame(
        columns=['Video_ID', 'Predicted_onset', 'Predicted_offset', 'Duration'])

    for index, row in df_pred.iterrows():
        if data_type == "CASME2":
            video_id = row['file_path'].split('/')[9][0:7]
            segment_num = row['file_path'].split('/')[10].split('.')[0]
        elif data_type == "SAMM":
            video_id = row['file_path'].split('/')[8]
            segment_num = row['file_path'].split('/')[9].split('.')[0]
        else:
            raise Exception("data_type should be either CASME2 or SAMM")

        if video_id == pre_video:  # the same VideoID
            subject = video_id.split("_")[0]
            strategy_num = strategy_list[subject]
            if pre_label == 0:
                if row[label_type] == 1:  # just changed from 0 to 1
                    onset_seg = int(segment_num)
                    offset_seg = int(segment_num)
                    pre_label = 1
            elif pre_label == 1:
                if row[label_type] == 0:
                    if count < strategy_num:        # change types:   1 -> 0,  10 -> 0, 100 -> 0
                        count = count + 1
                    else:                           # changed type:   1000 -> 0
                        offset_seg = int(segment_num) - strategy_num - 1
                        onset = onset_seg * frame_step + 1
                        offset = offset_seg * frame_step + frame_len
                        duration = offset - onset
                        data = dict({'Video_ID': video_id, 'Predicted_onset': onset, 'Predicted_offset': offset,
                                     'Duration': duration})
                        df_result = df_result.append(data, ignore_index=True)

                        # initiate all the counter
                        pre_label = 0
                        onset_seg = 0
                        offset_seg = 0
                        onset = 0
                        offset = 0
                        count = 0
                else:       # change types: 1 -> 1,  10 -> 1, 100 -> 1, 1000 -> 1
                    pre_label = 1
                    offset_seg = int(segment_num)
                    count = 0
        else:     # change to a new video id
            # initiate all the counter
            pre_video = video_id
            onset_seg = 0
            offset_seg = 0
            onset = 0
            offset = 0
            pre_label = 0
            count = 0

            if row[label_type] == 1:
                pre_label = 1
                onset_seg = int(segment_num)

    return df_result


def recall_precision_f1(a_clips, m_clips, n_clips):
    """
    :param a_clips: the number of true positive emotion segment
    :param m_clips: the number of ground truth emotion segment
    :param n_clips: the number of spotted emotion segment
    :return:
    """

    if m_clips != 0:
        recall = float(a_clips) / float(m_clips)  # A / M
    else:
        recall = 0
    if n_clips != 0:
        precision = float(a_clips) / float(n_clips)  # A / N
    else:
        precision = 0
    if (recall + precision) != 0:
        f1_score = 2 * recall * precision / (recall + precision)
    else:
        f1_score = 0

    return recall, precision, f1_score


def calc_iou(clip1, clip2):

    intersection = max(0, min(clip1[1], clip2[1]) - max(clip1[0], clip2[0])) + 1
    union = (clip1[1] - clip1[0] + 1) + (clip2[1] - clip2[0] + 1) - intersection

    if clip1[1] <= clip1[0] or clip2[1] <= clip2[0] or union <= 0:
        return 0.0
    else:
        return float(intersection) / float(union)


def count_pred_clips(df_label, df_result, duration_threshold, iou_threshold=0.5):

    macro_df_report = micro_df_report = pd.DataFrame(
        columns=['Video_ID', 'GT_onset', 'GT_offset', 'Predicted_onset', 'Predicted_offset', 'Result'])

    for lbs_index, lbs_row in df_label.iterrows():          # iterate over the labels
        for pred_index, pred_row in df_result.iterrows():           # iterate over the results
            if pred_row['Video_ID'] == lbs_row['Video_ID']:
                clip1 = [pred_row['Predicted_onset'], pred_row['Predicted_offset']]
                clip2 = [lbs_row['GT_onset'], lbs_row['GT_offset']]
                data = dict(
                    {'Video_ID': pred_row['Video_ID'], 'GT_onset': lbs_row['GT_onset'],
                     'GT_offset': lbs_row['GT_offset'],
                     'Predicted_onset': pred_row['Predicted_onset'],
                     'Predicted_offset': pred_row['Predicted_offset'], 'Result': 'TP'})

                if lbs_row['Duration'] >= duration_threshold and pred_row['Duration'] >= duration_threshold \
                        and calc_iou(clip1, clip2) >= iou_threshold:
                    macro_df_report = macro_df_report.append(data, ignore_index=True)
                elif lbs_row['Duration'] < duration_threshold and pred_row['Duration'] < duration_threshold \
                        and calc_iou(clip1, clip2) >= iou_threshold:
                    micro_df_report = micro_df_report.append(data, ignore_index=True)

    return macro_df_report, micro_df_report


def evaluate_logfile(macro_log_file_path, micro_log_file_path):

    # read log file and count the results
    def evaluate_single_log(log_file_path):
        # Initial flag and A, M, N
        flag = 0
        A_1 = 0
        M_1 = 0
        N_1 = 0
        A_2 = 0
        M_2 = 0
        N_2 = 0
        df_log = pd.read_csv(log_file_path)
        for index, row in df_log.iterrows():
            if row['Video_ID'] == '2.0':
                flag = 1

            # calculate A, M, N for CASME2
            if flag == 0 and row['Result'] == 'TP':
                A_1 = A_1 + 1
                M_1 = M_1 + 1
                N_1 = N_1 + 1
            if flag == 0 and row['Result'] == 'FN':
                M_1 = M_1 + 1
            if flag == 0 and row['Result'] == 'FP':
                N_1 = N_1 + 1

            # calculate A, M, N for SAMM
            if flag == 1 and row['Result'] == 'TP':
                A_2 = A_2 + 1
                M_2 = M_2 + 1
                N_2 = N_2 + 1
            if flag == 1 and row['Result'] == 'FN':
                M_2 = M_2 + 1
            if flag == 1 and row['Result'] == 'FP':
                N_2 = N_2 + 1

        return A_1, M_1, N_1, A_2, M_2, N_2

    # calculate the f1-scores for MaE and ME of each dataset
    MaE_A1, MaE_M1, MaE_N1, MaE_A2, MaE_M2, MaE_N2 = evaluate_single_log(macro_log_file_path)
    ME_A1, ME_M1, ME_N1, ME_A2, ME_M2, ME_N2 = evaluate_single_log(micro_log_file_path)
    print(MaE_A1, MaE_M1, MaE_N1)
    print(ME_A1, ME_M1, ME_N1)
    _, _, casme2_mae_f1 = recall_precision_f1(MaE_A1, MaE_M1, MaE_N1)
    _, _, casme2_me_f1 = recall_precision_f1(ME_A1, ME_M1, ME_N1)
    _, _, samm_mae_f1 = recall_precision_f1(MaE_A2, MaE_M2, MaE_N2)
    _, _, samm_me_f1 = recall_precision_f1(ME_A2, ME_M2, ME_N2)

    # calculate the overall f1-scores of each dataset
    overall_A1 = MaE_A1 + ME_A1
    overall_A2 = MaE_A2 + ME_A2
    overall_M1 = MaE_M1 + ME_M1
    overall_M2 = MaE_M2 + ME_M2
    overall_N1 = MaE_N1 + ME_N1
    overall_N2 = MaE_N2 + ME_N2
    _, _, overall_casme2_f1 = recall_precision_f1(overall_A1, overall_M1, overall_N1)
    _, _, overall_samm_f1 = recall_precision_f1(overall_A2, overall_M2, overall_N2)

    return casme2_mae_f1, casme2_me_f1, overall_casme2_f1, samm_mae_f1, samm_me_f1, overall_samm_f1
