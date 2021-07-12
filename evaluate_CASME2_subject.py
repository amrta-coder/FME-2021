import os
import pandas as pd
from evaluation import process_label_file, process_result_file, recall_precision_f1, count_pred_clips

subject_list = {'15': 0, '16': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0, '25': 0, '26': 0,
                '27': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0, '37': 0,
                '38': 0, '40': 0}


def calc_casme2_subject(result_file_path, subject, label_type):
    """
    :param result_filepath:
    :param subject: the subject number for a certain person
    :param label_type: "pred_threshold" or "pred_threshold_modify"
    :return: csv file to certain file path
    """
    strategy_report = pd.DataFrame(
        columns=['Strategy', 'Subject_ID', 'F1_Score', 'TP_Micro', 'TP_Macro',
                 'GT_Micro', 'GT_Macro', 'Pred_Micro', 'Pred_Macro', 'TP', 'FN', 'FP'])

    # calculate the Ground Truth number of micro- and macro-expression for certain subject
    GT_Micro_Num = 0
    GT_Macro_Num = 0
    for index, row in df_label.iterrows():
        if subject in row['Video_ID'] and row['Duration'] < 15:
            GT_Micro_Num = GT_Micro_Num + 1
        if subject in row['Video_ID'] and row['Duration'] >= 15:
            GT_Macro_Num = GT_Macro_Num + 1

    # process results
    for strategy in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        CASME2_result = pd.read_csv(result_file_path)
        df_result = process_result_file(CASME2_result, 8, 4, strategy, 'CASME2', label_type)

        # calculate the predicted micro- and macro-expression for certain subject
        pre_clip_num = 0
        pred_micro_num = 0
        pred_macro_num = 0
        for index, row in df_result.iterrows():
            if subject in row['Video_ID']:
                pre_clip_num = pre_clip_num + 1
                if row['Duration'] >= 15:
                    pred_macro_num = pred_macro_num + 1
                elif row['Duration'] < 15:
                    pred_micro_num = pred_micro_num + 1

        # calculate the true positive micro- and macro-expression for certain subject
        macro_df_report, micro_df_report = count_pred_clips(df_label, df_result, 15)

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
    label_file = "./labels/CAS(ME)^2code_final.xlsx"
    df_label = process_label_file(label_file)

    # process reports for each subject
    result_file_path = "results/20210706-093237_casme2_TextCNN32.csv"

    for threshold in ["pred_threshold_0.1", "pred_threshold_0.1_modify",
                      "pred_threshold_0.2", "pred_threshold_0.2_modify",
                      "pred_threshold_0.30000000000000004", "pred_threshold_0.30000000000000004_modify",
                      "pred_threshold_0.4", "pred_threshold_0.4_modify",
                      "pred_threshold_0.5", "pred_threshold_0.5_modify",
                      "pred_threshold_0.6", "pred_threshold_0.6_modify",
                      "pred_threshold_0.7000000000000001", "pred_threshold_0.7000000000000001_modify",
                      "pred_threshold_0.8", "pred_threshold_0.8_modify",
                      "pred_threshold_0.9", "pred_threshold_0.9_modify"]:
        for subject in subject_list:
            calc_casme2_subject(result_file_path, subject, threshold)

    # after terminated notification to slack
    import atexit
    from utils.connect_slack import send_message_to_slack

    atexit.register(send_message_to_slack, config_name=result_file_path)

