import pandas as pd
from evaluation import process_label_file, process_result_file, recall_precision_f1, count_pred_clips
import copy

subject_micro = {'006': 0, '007': 0, '008': 0, '009': 0, '010': 0, '011': 0, '012': 0, '013': 0, '014': 0, '015': 0,
                 '016': 0, '017': 0, '018': 0, '019': 0, '020': 0, '021': 0, '022': 0, '023': 0, '024': 0, '025': 0,
                 '026': 0, '028': 0, '030': 0, '031': 0, '032': 0, '033': 0, '034': 0, '035': 0, '036': 0, '037': 0}


if __name__ == "__main__":
    # process labels
    SAMM_label = "./labels/SAMM_LongVideos_V3_Release.xlsx"
    df_label = process_label_file(SAMM_label)

    strategy_report = pd.DataFrame(
        columns=['Strategy', 'Overall_F1-Score', 'All_pred_num', 'Each_Micro',
                 'TP_Micro_Number', 'Each_Macro', 'TP_Macro_Number'])

    # process results
    for strategy in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        SAMM_result = pd.read_csv("./results/result-6.csv")
        df_result = process_result_file(SAMM_result, 40, 20, strategy, 'SAMM', 'pred_threshold')
        macro_df_report, micro_df_report = count_pred_clips(df_label, df_result, 100)
        macro_num = len(macro_df_report)
        micro_num = len(micro_df_report)

        # calculate the f1_score
        lbs_clip_num = len(df_label) - 1
        pre_clip_num = len(df_result) - 1
        tp_num = len(macro_df_report) - 1 + len(micro_df_report) - 1
        recall, precision, f1_score = recall_precision_f1(tp_num, lbs_clip_num, pre_clip_num)

        # calculate the number of segment found for each subject
        dic_subject_micro = copy.deepcopy(subject_micro)
        dic_subject_macro = copy.deepcopy(subject_micro)
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

    strategy_report.to_excel('./evaluations/result-6.xlsx',
                             sheet_name='result-6', index=False)
