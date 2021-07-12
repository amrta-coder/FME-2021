import os
import pandas as pd
import copy
from evaluation import process_label_file, process_result_file, calc_iou, process_result_file_best_strategy

# define the strategy report
log_file = pd.DataFrame(
    columns=['Video_ID', 'GT_onset', 'GT_offset', 'Predicted_onset', 'Predicted_offset', 'Result'])


def generate_df_log(df_label, df_result, dataset_type="CASME2"):

    if dataset_type is "SAMM":
        duration_threshold = 100
    elif dataset_type is "CASME2":
        duration_threshold = 15
    iou_threshold = 0.5
    macro_df_log = micro_df_log = copy.deepcopy(log_file)

    # find true positive of
    for lbs_index, lbs_row in df_label.iterrows():  # iterate over the labels
        for pred_index, pred_row in df_result.iterrows():  # iterate over the results
            if pred_row['Video_ID'] == lbs_row['Video_ID']:
                clip1 = [pred_row['Predicted_onset'], pred_row['Predicted_offset']]
                clip2 = [lbs_row['GT_onset'], lbs_row['GT_offset']]
                data = dict({'Video_ID': pred_row['Video_ID'],
                             'GT_onset': lbs_row['GT_onset'], 'GT_offset': lbs_row['GT_offset'],
                             'Predicted_onset': pred_row['Predicted_onset'],
                             'Predicted_offset': pred_row['Predicted_offset'], 'Result': 'TP'})
                if lbs_row['Duration'] >= duration_threshold and pred_row['Duration'] >= duration_threshold \
                        and calc_iou(clip1, clip2) >= iou_threshold:
                    macro_df_log = macro_df_log.append(data, ignore_index=True)
                    df_label.drop(index=lbs_index, inplace=True)
                    df_result.drop(index=pred_index, inplace=True)
                elif lbs_row['Duration'] < duration_threshold and pred_row['Duration'] < duration_threshold \
                        and calc_iou(clip1, clip2) >= iou_threshold:
                    micro_df_log = micro_df_log.append(data, ignore_index=True)
                    df_label.drop(index=lbs_index, inplace=True)
                    df_result.drop(index=pred_index, inplace=True)

    # process the left record for label file
    for lbs_index, lbs_row in df_label.iterrows():
        if lbs_row['Duration'] >= duration_threshold:
            data = dict({'Video_ID': lbs_row['Video_ID'], 'GT_onset': lbs_row['GT_onset'],
                         'GT_offset': lbs_row['GT_offset'], 'Predicted_onset': "-",
                         'Predicted_offset': "-", 'Result': 'FN'})
            macro_df_log = macro_df_log.append(data, ignore_index=True)
        elif lbs_row['Duration'] < duration_threshold:
            data = dict({'Video_ID': lbs_row['Video_ID'], 'GT_onset': lbs_row['GT_onset'],
                         'GT_offset': lbs_row['GT_offset'], 'Predicted_onset': "-",
                         'Predicted_offset': "-", 'Result': 'FN'})
            micro_df_log = micro_df_log.append(data, ignore_index=True)

    # process the left record for result file
    for pred_index, pred_row in df_result.iterrows():
        if pred_row['Duration'] >= duration_threshold:
            data = dict({'Video_ID': pred_row['Video_ID'], 'GT_onset': "-",
                         'GT_offset': "-", 'Predicted_onset': pred_row['Predicted_onset'],
                         'Predicted_offset': pred_row['Predicted_offset'], 'Result': 'FP'})
            macro_df_log = macro_df_log.append(data, ignore_index=True)
        elif pred_row['Duration'] < duration_threshold:
            data = dict({'Video_ID': pred_row['Video_ID'], 'GT_onset': "-",
                         'GT_offset': "-", 'Predicted_onset': pred_row['Predicted_onset'],
                         'Predicted_offset': pred_row['Predicted_offset'], 'Result': 'FP'})
            micro_df_log = micro_df_log.append(data, ignore_index=True)

    return micro_df_log, macro_df_log


def generate_log(CASME2_label, SAMM_label, CASME2_results, SAMM_results, is_best_strategy=False, **kw):

    if 'CASME2_strategy' in kw:
        CASME2_strategy = kw['CASME2_strategy']
    if 'SAMM_strategy' in kw:
        SAMM_strategy = kw['SAMM_strategy']
    if 'CASME2_strategy_folder' in kw:
        CASME2_strategy_folder = kw['CASME2_strategy_folder']
    if 'SAMM_strategy_folder' in kw:
        SAMM_strategy_folder = kw['SAMM_strategy_folder']
    if 'CASME2_label_type' in kw:
        CASME2_label_type = kw['CASME2_label_type']
    if 'SAMM_label_type' in kw:
        SAMM_label_type = kw['SAMM_label_type']

    print(CASME2_label_type)
    print(SAMM_label_type)

    # process labels of CAS(ME)2 and SAMM
    df_casme2_label = process_label_file(CASME2_label)
    df_samm_label = process_label_file(SAMM_label)

    # process result file of CAS(ME)2 and SAMM
    if not is_best_strategy:
        df_casme2_result = process_result_file(CASME2_results, 8, 4,
                                               CASME2_strategy, "CASME2", CASME2_label_type)
        df_samm_result = process_result_file(SAMM_results, 40, 20,
                                             SAMM_strategy, "SAMM", SAMM_label_type)
    else:
        df_casme2_result = process_result_file_best_strategy(CASME2_results, 8, 4, CASME2_strategy_folder,
                                                             "CASME2", CASME2_label_type)
        df_samm_result = process_result_file_best_strategy(SAMM_results, 40, 20, SAMM_strategy_folder,
                                                           "SAMM", SAMM_label_type)

    casme2_micro_df_log, casme2_macro_df_log = generate_df_log(df_casme2_label, df_casme2_result, "CASME2")
    samm_micro_df_log, samm_macro_df_log = generate_df_log(df_samm_label, df_samm_result, "SAMM")

    data1 = dict({'Video_ID': int(1)})
    data2 = dict({'Video_ID': int(2)})

    # micro_log_list = [data1, casme2_micro_df_log, data2, samm_micro_df_log]
    # macro_log_list = [data1, casme2_macro_df_log, data2, samm_macro_df_log]
    casme2_micro_df_log.sort_values(["Video_ID"], inplace=True)
    casme2_macro_df_log.sort_values(["Video_ID"], inplace=True)
    samm_micro_df_log.sort_values(["Video_ID"], inplace=True)
    samm_macro_df_log.sort_values(["Video_ID"], inplace=True)

    macro_log_1 = micro_log_1 = copy.deepcopy(log_file)
    macro_log_2 = micro_log_2 = copy.deepcopy(log_file)
    data_name_1 = dict({'Video_ID': 1})
    data_name_2 = dict({'Video_ID': 2})
    macro_log_1 = macro_log_1.append(data_name_1, ignore_index=True)
    micro_log_1 = micro_log_1.append(data_name_1, ignore_index=True)
    macro_log_2 = macro_log_2.append(data_name_2, ignore_index=True)
    micro_log_2 = micro_log_2.append(data_name_2, ignore_index=True)

    micro_log_list = [micro_log_1, casme2_micro_df_log, micro_log_2, samm_micro_df_log]
    macro_log_list = [macro_log_1, casme2_macro_df_log, macro_log_2, samm_macro_df_log]
    micro_log = pd.concat(micro_log_list)
    macro_log = pd.concat(macro_log_list)

    micro_log.to_csv("submit_logs/micro_log.csv", index=False)
    macro_log.to_csv("submit_logs/macro_log.csv", index=False)


if __name__ == "__main__":

    directory = "submit_logs"
    if not os.path.exists(directory):
        os.mkdir(directory)

    # designate labels of CAS(ME)2 and SAMM
    CASME2_label = "./labels/CAS(ME)^2code_final.xlsx"
    SAMM_label = "./labels/SAMM_LongVideos_V3_Release.xlsx"

    # designate result file of CAS(ME)2 and SAMM
    CASME2_results = pd.read_csv("results/20210706-093237_casme2_TextCNN32.csv")
    SAMM_results = pd.read_csv("results/20210625-094832_samm_GRU64.csv")

    # designate the strategy folder of CAS(ME)2 and SAMM
    CASME2_strategy_folder = "./evaluations/20210706-093237_casme2_TextCNN32_pred_threshold_modify"
    CASME2_label_type = "pred_threshold_modify"

    SAMM_strategy_folder = "./evaluations/20210625-094832_samm_GRU64-pred_threshold"
    SAMM_label_type = "pred_threshold"

    generate_log(CASME2_label, SAMM_label, CASME2_results, SAMM_results, True,
                 CASME2_strategy_folder=CASME2_strategy_folder, SAMM_strategy_folder=SAMM_strategy_folder,
                 CASME2_label_type=CASME2_label_type, SAMM_label_type=SAMM_label_type)
