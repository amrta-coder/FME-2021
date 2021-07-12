import os
import pandas as pd
from evaluation import process_label_file, process_result_file, recall_precision_f1, count_pred_clips

import atexit
from utils.connect_slack import send_message_to_slack

subject_list = {'006': 0, '007': 0, '008': 0, '009': 0, '010': 0, '011': 0, '012': 0, '013': 0, '014': 0, '015': 0,
                '016': 0, '017': 0, '018': 0, '019': 0, '020': 0, '021': 0, '022': 0, '023': 0, '024': 0, '025': 0,
                '026': 0, '028': 0, '030': 0, '031': 0, '032': 0, '033': 0, '034': 0, '035': 0, '036': 0, '037': 0}


def calc_samm_subject(result_file_path, subject, label_type):
    """
    :param result_filepath:
    :param subject: the subject number for a certain person
    :param label_type: "pred_threshold" or "pred_threshold_modify"
    :return: csv file to certain file path
    """

    # define the strategy report
    strategy_report = pd.DataFrame(
        columns=['Strategy', 'Subject_ID', 'F1_Score', 'TP_Micro', 'TP_Macro',
                 'GT_Micro', 'GT_Macro', 'Pred_Micro', 'Pred_Macro', 'TP', 'FN', 'FP'])

    # calculate the Ground Truth number of micro- and macro-expression for certain subject
    GT_Micro_Num = 0
    GT_Macro_Num = 0
    for index, row in df_label.iterrows():
        if subject in row['Video_ID'] and row['Duration'] < 100:
            GT_Micro_Num = GT_Micro_Num + 1
        if subject in row['Video_ID'] and row['Duration'] >= 100:
            GT_Macro_Num = GT_Macro_Num + 1

    # process results
    for strategy in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        SAMM_result = pd.read_csv(result_file_path)
        df_result = process_result_file(SAMM_result, 40, 20, strategy, 'SAMM', label_type)

        # calculate the predicted micro- and macro-expression for certain subject
        pre_clip_num = 0
        pred_micro_num = 0
        pred_macro_num = 0
        for index, row in df_result.iterrows():
            if subject in row['Video_ID']:
                pre_clip_num = pre_clip_num + 1
                if row['Duration'] >= 100:
                    pred_macro_num = pred_macro_num + 1
                elif row['Duration'] < 100:
                    pred_micro_num = pred_micro_num + 1

        # calculate the true positive micro- and macro-expression for certain subject
        macro_df_report, micro_df_report = count_pred_clips(df_label, df_result, 100)

        macro_num = 0
        for index, row in macro_df_report.iterrows():
            if subject in row['Video_ID']:
                macro_num = macro_num + 1

        micro_num = 0
        for index, row in micro_df_report.iterrows():
            if subject in row['Video_ID']:
                micro_num = micro_num + 1

        # calculate recall, precision, f1-score for certain subject
        lbs_clip_num = GT_Micro_Num + GT_Macro_Num
        tp_num = macro_num + micro_num
        recall, precision, f1_score = recall_precision_f1(tp_num, lbs_clip_num, pre_clip_num)

        tp = micro_num + macro_num
        fn = GT_Micro_Num + GT_Macro_Num - tp
        fp = pred_micro_num + pred_macro_num - tp
        data = dict(
            {'Strategy': strategy, 'Subject_ID': subject, 'F1_Score': f1_score,
             'TP_Micro': micro_num, 'TP_Macro': macro_num,
             'GT_Micro': GT_Micro_Num, 'GT_Macro': GT_Macro_Num,
             'Pred_Micro': pred_micro_num, 'Pred_Macro': pred_macro_num,
             'TP': tp, 'FN': fn, 'FP': fp})
        strategy_report = strategy_report.append(data, ignore_index=True)

    # save the report to certain path with certain names of csv file
    directory = './evaluations/' + result_file_path.split("/")[1].split(".")[0] + "-" + label_type + "/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    report_file_path = directory + subject + "-report.csv"
    strategy_report.to_csv(report_file_path, index=False)

    # save the report to certain path with certain names of excel file
    # sheet_name = subject + "-report.csv"
    # strategy_report.to_excel(result_file_path, sheet_name, index=False)


if __name__ == "__main__":

    # process labels
    SAMM_label = "./labels/SAMM_LongVideos_V3_Release.xlsx"
    df_label = process_label_file(SAMM_label)

    # process reports for each subject
    result_file_path = "results/20210625-094832_samm_GRU64.csv"
    for threshold in ["pred_threshold_0.30000000000000004", "pred_threshold_0.30000000000000004_modify",
                      "pred_threshold_0.4", "pred_threshold_0.4_modify",
                      "pred_threshold_0.5", "pred_threshold_0.5_modify",
                      "pred_threshold_0.6", "pred_threshold_0.6_modify",
                      "pred_threshold_0.7000000000000001", "pred_threshold_0.7000000000000001_modify",
                      "pred_threshold_0.8", "pred_threshold_0.8_modify",
                      "pred_threshold_0.9", "pred_threshold_0.9_modify"]:
        for subject in subject_list:
            calc_samm_subject(result_file_path, subject, threshold)

    # after terminated notification to slack
    atexit.register(send_message_to_slack, config_name=result_file_path)
