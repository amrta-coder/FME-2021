import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
from skimage import exposure
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatruesSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, batch_size=16, verbose=0, shuffle_on_epoch_end=True,
                 random_state=1, test=False, n_timesteps=180):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param batch_size: int
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.test = test
        self.n_timesteps = n_timesteps
        self.prepare_dataset()
        self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))


    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_au_csv(x_path) for x_path in batch_x_path])
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_au_csv(self, openface_file):
        features_array = pd.read_csv(openface_file).values.astype('float32')
        if features_array.shape[0] == 0:
            print(f'-----------------Error:{openface_file} size is 0!!!')
        else:
            stdsc = StandardScaler()
            # stdsc = MinMaxScaler()
            features_array = stdsc.fit_transform(features_array)
            return features_array

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size]

    def prepare_dataset(self):
        if self.test:
            df = self.dataset_df
        else:
            df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y = df["file_path"].values, df[["frame_start", "frame_end"]].values.astype('float32') / self.n_timesteps

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
