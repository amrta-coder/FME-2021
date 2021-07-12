import numpy as np
import os
from configparser import ConfigParser
from generator_vad import FeatruesSequence
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import get_optimal_precision_recall, pred_modify
from model import Conv1D_model, LSTM_model, GRU_model, ConcatCNN_model
from keras import backend as K


def set_sess_cfg():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    # test config
    output_dir_samm = cp["TEST"].get("output_dir_samm")
    batch_size = cp["TEST"].getint("batch_size")
    # parse weights file path
    output_model_name = cp["TRAIN"].get("output_model_name")
    fme_dir_samm = cp["DEFAULT"].get("fme_dir_samm")
    features_engineered = cp["DEFAULT"].get("features_engineered")
    label_samm_predict = cp["TEST"].get("label_samm_predict")
    model_name = cp["TEST"].get("model_name")
    n_timesteps = cp["DEFAULT"].getint("n_timesteps_samm")
    n_features = cp["DEFAULT"].getint("n_features_vad")
    use_weights = cp["TEST"].getboolean("use_weights")
    test_csv = os.path.join(fme_dir_samm, features_engineered, label_samm_predict)

    threshold_max_list = []
    f1_score_max_list = []
    precision_max_list = []
    recall_max_list = []
    list_subject = list(range(6, 38))
    list_subject.remove(27)
    list_subject.remove(29)
    for subject in list_subject:
        print(f'subdir is : {subject}')
        K.clear_session()
        output_samm_subdir = os.path.join(output_dir_samm, str(subject))
        # predict_csv = os.path.join(output_samm_subdir, 'prediction.csv')
        # df_test_csv = pd.read_csv(test_csv)
        # df_test_csv['subject'] = df_test_csv['file_path'].apply(lambda x: int(x.split('/')[8].split('_')[0]))
        # df_selected = df_test_csv.loc[df_test_csv['subject'] == subject]
        # df_selected[['file_path', 'label']].to_csv(predict_csv, index=False)
        test_sequence = FeatruesSequence(
            dataset_csv_file=test_csv,
            batch_size=batch_size,
            shuffle_on_epoch_end=False,
            test=True
        )
        model_path = os.path.join(output_samm_subdir, output_model_name)

        print("** load model **")
        # model_method = eval(f'{model_name}_model')
        # model = model_method(n_timesteps=n_timesteps, n_features=n_features)
        if use_weights:
            model_method = eval(f'{model_name}_model')
            model = model_method(n_timesteps=n_timesteps, n_features=n_features)
            model.load_weights(model_path)
        else:
            model = load_model(model_path)

        print("** make prediction **")
        # model_train.compile(optimizer=Adam(), loss="mean_squared_error")
        prob_array = model.predict_generator(test_sequence,
                                                   # steps=test_steps,
                                                   max_queue_size=8,
                                                   workers=8,
                                                   use_multiprocessing=True,
                                                   verbose=1
                                                   )
        df_test = pd.read_csv(test_csv)
        df_test['probs'] = prob_array
        y_true = df_test['label']
        threshold_max, f1_score_max, precision_max, recall_max = get_optimal_precision_recall(y_true, prob_array)
        print(f'threshold : {threshold_max},precision_max : {precision_max},recall_max : {recall_max},f1_score_max : {f1_score_max}')
        pred_threshold = (prob_array > threshold_max).astype(int)
        df_test['pred_threshold'] = pred_threshold
        df_test['pred_threshold_modify'] = pred_modify(pred_threshold)
        for thresh in np.linspace(start=0.1, stop=0.9, num=9):
            pred_threshold = (prob_array > thresh).astype(int)
            df_test[f'pred_threshold_{str(thresh)}'] = pred_threshold
            df_test[f'pred_threshold_{str(thresh)}_modify'] = pred_modify(pred_threshold)

        df_test.to_csv(os.path.join(output_samm_subdir, 'result.csv'), index=False)
        evaluate_file = os.path.join(output_samm_subdir, 'threshold.log')
        threshold_max_list.append(threshold_max)
        precision_max_list.append(precision_max)
        recall_max_list.append(recall_max)
        f1_score_max_list.append(f1_score_max)
        with open(str(evaluate_file), 'w') as f:
            f.write(f'threshold : {threshold_max} \n')
            f.write(f'precision_max : {precision_max} \n')
            f.write(f'recall_max : {recall_max} \n')
            f.write(f'f1_score_max : {f1_score_max} \n')
    df_results = pd.DataFrame(
        {'subject': list_subject, 'threshold_max_list': threshold_max_list, 'precision_max_list': precision_max_list,
         'recall_max_list': recall_max_list, 'f1_score_max_list': f1_score_max_list})
    df_results.to_csv(os.path.join(output_dir_samm, 'best_f1.csv'), index=False)


if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
