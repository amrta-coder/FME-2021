import numpy as np
import os
from configparser import ConfigParser
from generator_vad import FeatruesSequence
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import get_optimal_precision_recall, pred_modify


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
    model_path = os.path.join(output_dir_samm, output_model_name)
    fme_dir_samm = cp["DEFAULT"].get("fme_dir_samm")
    features_engineered = cp["DEFAULT"].get("features_engineered")
    model_name = cp["TEST"].get("model_name")
    n_timesteps = cp["DEFAULT"].getint("n_timesteps_samm")
    n_features = cp["DEFAULT"].getint("n_features_vad")
    use_weights = cp["TEST"].getboolean("use_weights")
    label_samm_predict_window = cp["TEST"].get("label_samm_predict_window")
    test_csv = os.path.join(fme_dir_samm, features_engineered, label_samm_predict_window)

    print("** load model **")
    if use_weights:
        model_method = eval(f'{model_name}_model')
        model = model_method(n_timesteps=n_timesteps, n_features=n_features)
        model.load_weights(model_path)
    else:
        model = load_model(model_path)

    print("** load test generator **")
    test_sequence = FeatruesSequence(
        dataset_csv_file=test_csv,
        batch_size=batch_size,
        shuffle_on_epoch_end=False,
        test=True
    )

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
    threshold, f1_score_max, precision_max, recall_max = get_optimal_precision_recall(y_true, prob_array)
    print(f'threshold : {threshold},precision_max : {precision_max},recall_max : {recall_max},f1_score_max : {f1_score_max}')
    pred_threshold = (prob_array > threshold).astype(int)
    df_test['pred_threshold'] = pred_threshold
    df_test['pred_threshold_modify'] = pred_modify(pred_threshold)
    for thresh in np.linspace(start=0.1, stop=0.9, num=9):
        pred_threshold = (prob_array > thresh).astype(int)
        df_test[f'pred_threshold_{str(thresh)}'] = pred_threshold
        df_test[f'pred_threshold_{str(thresh)}_modify'] = pred_modify(pred_threshold)

    df_test.to_csv(os.path.join(output_dir_samm, 'result.csv'), index=False)
    evaluate_file = os.path.join(output_dir_samm, 'threshold.log')
    with open(str(evaluate_file), 'w') as f:
        f.write(f'threshold : {threshold} \n')
        f.write(f'precision_max : {precision_max} \n')
        f.write(f'recall_max : {recall_max} \n')
        f.write(f'f1_score_max : {f1_score_max} \n')


if __name__ == "__main__":
    np.set_printoptions(threshold=1e6)
    set_sess_cfg()
    main()
