import pandas as pd
from evaluation import process_label_file, process_result_file, recall_precision_f1, count_pred_clips
import copy

subject_list = {'15': 0, '16': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0, '25': 0, '26': 0,
                 '27': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0, '37': 0,
                 '38': 0, '40': 0}


if __name__ == "__main__":
    # process labels
    label_file = "./labels/CAS(ME)^2code_final.xlsx"
    df_label = process_label_file(label_file)

    strategy_report = pd.DataFrame(
        columns=['Strategy', 'Overall_F1-Score', 'All_pred_num', 'Each_Micro',
                 'TP_Micro_Number', 'Each_Macro', 'TP_Macro_Number'])

    # process results
    for strategy in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        df_pred = pd.read_csv("./results/result-20210619-094552.csv")
        df_result = process_result_file(df_pred, 8, 4, strategy, 'CASME2')
        macro_df_report, micro_df_report = count_pred_clips(df_label, df_result, 15)
        macro_num = len(macro_df_report)
        micro_num = len(micro_df_report)

        # calculate the f1_score
        lbs_clip_num = len(df_label) - 1
        pre_clip_num = len(df_result) - 1
        tp_num = len(macro_df_report) - 1 + len(micro_df_report) - 1
        recall, precision, f1_score = recall_precision_f1(tp_num, lbs_clip_num, pre_clip_num)

        # calculate the number of segment found for each subject
        dic_subject_micro = copy.deepcopy(subject_list)
        dic_subject_macro = copy.deepcopy(subject_list)
        for index, row in micro_df_report.iterrows():
            subject = row['Video_ID'].split('_')[0]
            dic_subject_micro[subject] = dic_subject_micro[subject] + 1

        for index, row in macro_df_report.iterrows():
            subject = row['Video_ID'].split('_')[0]
            dic_subject_macro[subject] = dic_subject_macro[subject] + 1

        data = dict(
            {'Strategy': strategy, 'Overall_F1-Score': f1_score, 'All_pred_num': pre_clip_num,
             'Each_Micro': dic_subject_micro, 'TP_Micro_Number': micro_num,
             'Each_Macro': dic_subject_macro, 'TP_Macro_Number': macro_num})
        strategy_report = strategy_report.append(data, ignore_index=True)
        strategy_report.to_excel('./evaluations/result-20210619-094552.xlsx',
                                 sheet_name='result-20210619-094552', index=False)
