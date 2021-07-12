import pandas as pd
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser
from generator_vad import FeatruesSequence
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, StratifiedKFold
import shutil
from datetime import datetime
from model import Conv1D_model, LSTM_model, GRU_model, ConcatCNN_model
from clr_callback import *
from callback import SaveMinLoss
import pickle
from sklearn.preprocessing import StandardScaler
from math import ceil
from pathlib import Path
from keras.utils import multi_gpu_model, plot_model
from model_resnet1d import Resnet1D
import tensorflow as tf


def set_sess_cfg():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def save_config(output):
    if not os.path.exists(output):
        os.makedirs(output)
    dir = os.path.dirname(__file__)
    output = os.path.join(output, str(datetime.now().strftime("%Y%m%d-%H%M%S")+'_casme2_uncrop_'+model_name+str(n_gru_units)))
    output_dir_src = os.path.join(output, 'src')
    os.makedirs(output_dir_src, exist_ok=True)
    print(f"backup config file to {output_dir_src}")
    shutil.copy("config.ini", os.path.join(output_dir_src, 'config.ini'))
    shutil.copy("model.py", os.path.join(output_dir_src, 'model.py'))
    train_file = os.path.basename(__file__)
    shutil.copy(os.path.join(dir, train_file), os.path.join(output_dir_src, train_file))
    return output


def scheduler(epoch, lr):
    if epoch < 10:
        lr = lr*0.1
    elif epoch < 20:
        lr = lr*0.01
    elif epoch < 30:
        lr = lr*0.001
    elif epoch < 40:
        lr = lr*0.0001
    return lr

def main(output_dir, optimizer):
    model_method = eval(f'{model_name}_model')
    if model_name == 'ConcatCNN':
        model = model_method(n_timesteps=n_timesteps, n_features=n_features)
    else:
        model = model_method(n_timesteps=n_timesteps, n_features=n_features, n_gru_units=n_gru_units)
    print(model.summary())

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
    X = df_label['file_path']
    Y = df_label['label']
    best_train_mses = []
    best_val_mses = []
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        # del model
        K.clear_session()
        # gc.collect()
        output_dir_full = os.path.join(output_dir, str(i))
        os.makedirs(output_dir_full, exist_ok=True)
        # get train/dev sample counts
        train_counts = len(train_index)
        val_counts = len(test_index)
        train_steps = ceil(train_counts / batch_size)
        validation_steps = ceil(val_counts / batch_size)

        train_csv = Path(output_dir_full).joinpath("train_{:02d}.csv".format(i))
        val_csv = Path(output_dir_full).joinpath("val_{:02d}.csv".format(i))
        df_label.loc[train_index].to_csv(str(train_csv), index=False, columns=["file_path", "label"])
        df_label.loc[test_index].to_csv(str(val_csv), index=False, columns=["file_path", "label"])

        train_sequence = FeatruesSequence(
                dataset_csv_file=str(train_csv),
                # csv_source_dir=csv_source_dir,
                batch_size=batch_size,
                random_state=seed,
            )
        validation_sequence = FeatruesSequence(
            dataset_csv_file=str(val_csv),
            # csv_source_dir=csv_source_dir,
            batch_size=batch_size,
            shuffle_on_epoch_end=False,
            test=True,
        )
        plot_model(model, to_file=os.path.join(output_dir_full, 'model.png'))
        if optimizer == 'adam':
            optimizer = Adam(lr=initial_learning_rate)
            base_lr = base_lr_adam
            max_lr = max_lr_adam
        elif optimizer == 'sgd':
            optimizer = SGD(momentum=momentum, decay=decay)
            base_lr = base_lr_sgd
            max_lr = max_lr_sgd
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['mse', 'mae'])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=36, verbose=0, mode='min')

        # if cyclicLR_mode == 'exp_range':
        #     gamma = 0.99994
        # else:
        #     gamma = 1.
        # clr = CyclicLR(mode=cyclicLR_mode, step_size=train_steps*2, base_lr=base_lr, max_lr=max_lr, gamma=gamma)
        # change_lr = LearningRateScheduler(scheduler, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1)
        model_save_path = os.path.join(output_dir_full, output_model_name)
        checkpoint = ModelCheckpoint(
            model_save_path,
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
        )
        csv_logger = CSVLogger(os.path.join(output_dir_full, 'training.csv'))
        save_min_loss = SaveMinLoss(filepath=output_dir_full)
        tensor_board = TensorBoard(log_dir=os.path.join(output_dir_full, "logs"), batch_size=batch_size)
        callbacks = [
            checkpoint,
            tensor_board,
            csv_logger,
            # clr,
            reduce_lr,
            save_min_loss,
            earlystop,
        ]
        print("** start training **")
        # Training.
        history = model.fit_generator(
            generator=train_sequence,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_sequence,
            validation_steps=validation_steps,
            callbacks=callbacks,
            # class_weight='auto',
            workers=generator_workers,
            shuffle=False,
        )
        # dump history
        print("** dump history **")
        with open(os.path.join(output_dir_full, "history.pkl"), "wb") as f:
            pickle.dump({
                "history": history.history,
            }, f)
        print("** done! **")

        mse = history.history["mse"]
        val_mse = history.history["val_mse"]
        best_val_mses.append(min(val_mse))
        best_train_mses.append(min(mse))

        # h = clr.history
        # lr = h['lr']
        # iterations = h['iterations']
        # plt.xlabel('Training Iterations')
        # plt.ylabel('Learning Rate')
        # plt.title("CLR - 'exp_range' Policy")
        # plt.plot(iterations, lr)
        # plt.savefig(os.path.join(output_dir_full, 'iterations_lr.png'))
    results_mean = np.array(best_val_mses).mean()
    results_std = np.array(best_val_mses).std()
    print(f'best_train_mses:{best_train_mses}')
    print(f'best_val_mses:{best_val_mses}')
    print(f'best_val_mses mean:{results_mean}')
    print(f'best_val_mses std:{results_std}')
    results_file = os.path.join(output_dir, "results.log")
    with open(results_file, 'a') as f:
        f.write(f'best_train_mses:{best_train_mses}')
        f.write(f'best_val_mses:{best_val_mses}')
        f.write(f'best_val_mses mean:{results_mean}')
        f.write(f'best_val_mses std:{results_std}')

if __name__ == "__main__":
    set_sess_cfg()
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)
    fme_dir = cp["DEFAULT"].get("fme_dir")
    features_engineered = cp["DEFAULT"].get("features_engineered")
    features_subdir_vad = cp["DEFAULT"].get("features_subdir_vad")
    label_croped_file_vad = cp["DEFAULT"].get("label_croped_file_vad")
    batch_size = cp["TRAIN"].getint("batch_size")
    seed = cp["TRAIN"].getint("seed")
    output_fold = cp["DEFAULT"].get("output_fold")
    n_timesteps = cp["DEFAULT"].getint("n_timesteps_vad")
    n_features = cp["DEFAULT"].getint("n_features_vad")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    output_model_name = cp["TRAIN"].get("output_model_name")
    base_lr_adam = cp["TRAIN"].getfloat("base_lr_adam")
    max_lr_adam = cp["TRAIN"].getfloat("max_lr_adam")
    momentum = cp["TRAIN"].getfloat("momentum")
    decay = cp["TRAIN"].getfloat("decay")
    base_lr_sgd = cp["TRAIN"].getfloat("base_lr_sgd")
    max_lr_sgd = cp["TRAIN"].getfloat("max_lr_sgd")
    loss_func = cp["TRAIN"].get("loss_func")
    cyclicLR_mode = cp["TRAIN"].get("cyclicLR_mode")
    epochs = cp["TRAIN"].getint("epochs")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    optimizer = cp["TRAIN"].get("optimizer")
    model_name = cp["TRAIN"].get("model_name")
    n_gru_units = cp["TRAIN"].getint("n_gru_units")
    features_engineered_root = os.path.join(fme_dir, features_engineered)
    features_engineered_dir = os.path.join(features_engineered_root, features_subdir_vad)
    df_label = pd.read_csv(os.path.join(features_engineered_root, label_croped_file_vad))
    output_dir = save_config(output_fold)
    main(output_dir, optimizer)
