import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout, SpatialDropout1D, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, GlobalMaxPooling1D, Bidirectional, LSTM, GRU, Input
from keras.layers.merge import concatenate

# n_timesteps=180
# n_features=69
n_timesteps=8
n_features=17
#['AU01_r_std', 'AU02_r_std', 'AU04_r_std', 'AU05_r_std', 'AU06_r_std', 'AU07_r_std', 'AU09_r_std', 'AU10_r_std', 'AU12_r_std', 'AU14_r_std', 'AU15_r_std', 'AU17_r_std', 'AU20_r_std', 'AU23_r_std', 'AU25_r_std', 'AU26_r_std', 'AU45_r_std', 'AU01_r_variation', 'AU02_r_variation', 'AU04_r_variation', 'AU05_r_variation', 'AU06_r_variation', 'AU07_r_variation', 'AU09_r_variation', 'AU10_r_variation', 'AU12_r_variation', 'AU14_r_variation', 'AU15_r_variation', 'AU17_r_variation', 'AU20_r_variation', 'AU23_r_variation', 'AU25_r_variation', 'AU26_r_variation', 'AU45_r_variation', 'AU01_r_max', 'AU02_r_max', 'AU04_r_max', 'AU05_r_max', 'AU06_r_max', 'AU07_r_max', 'AU09_r_max', 'AU10_r_max', 'AU12_r_max', 'AU14_r_max', 'AU15_r_max', 'AU17_r_max', 'AU20_r_max', 'AU23_r_max', 'AU25_r_max', 'AU26_r_max', 'AU45_r_max', 'AU01_c_frequency', 'AU02_c_frequency', 'AU04_c_frequency', 'AU05_c_frequency', 'AU06_c_frequency', 'AU07_c_frequency', 'AU09_c_frequency', 'AU10_c_frequency', 'AU12_c_frequency', 'AU14_c_frequency', 'AU15_c_frequency', 'AU17_c_frequency', 'AU20_c_frequency', 'AU23_c_frequency', 'AU25_c_frequency', 'AU26_c_frequency', 'AU28_c_frequency', 'AU45_c_frequency']
def Conv1D_model(n_timesteps, n_features, n_outputs=1):
    model = Sequential()
    model.add(Conv1D(64, 7, input_shape=(n_timesteps, n_features), activation='relu', padding='same', name='con1'))
    model.add(MaxPooling1D(pool_size=2, padding='same', strides=2, name='mp1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(Conv1D(128, 7, padding='same', activation='relu', name='con2'))
    model.add(MaxPooling1D(pool_size=2, padding='same', strides=2, name='mp2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(Conv1D(256, 7, padding='same', activation='relu', name='con3'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='mp3'))
    # model.add(Conv1D(2, 5, padding='same', activation='relu'))
    # model.add(LSTM(32))
    model.add(GlobalAveragePooling1D(name='gap'))
    # model.add(GlobalMaxPooling1D())
    # model.add(Flatten())
    model.add(Dense(512, activation='relu', name='fc'))
    model.add(Dropout(0.5, name='drop'))
    model.add(Dense(n_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(0.05), name='output'))
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['mse'])
    return model

def ConcatCNN_model(n_timesteps, n_features, n_outputs=1):
    inputs = Input(shape=(n_timesteps, n_features))
    # 卷积层和池化层，设置卷积核大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu', name='con1')(inputs)
    cnn1 = MaxPooling1D(pool_size=8, name='mp1')(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu', name='con2')(inputs)
    cnn2 = MaxPooling1D(pool_size=4, name='mp2')(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu', name='con3')(inputs)
    cnn3 = MaxPooling1D(pool_size=2, name='mp3')(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = Flatten(name='flat')(cnn)
    #加上dropout防止过拟合
    drop = Dropout(0.5, name='drop')(flat)
    main_output = Dense(n_outputs, activation='sigmoid', name='output')(drop)
    model = Model(inputs=inputs, outputs=main_output)
    return model

def ConcatCNN_SAMM_model(n_timesteps, n_features, n_outputs=1):
    inputs = Input(shape=(n_timesteps, n_features))
    # 卷积层和池化层，设置卷积核大小分别为3,4,5
    cnn1 = Conv1D(256, 5, padding='same', strides=2, activation='relu', name='con1')(inputs)
    cnn1 = MaxPooling1D(pool_size=8, name='mp1')(cnn1)
    cnn2 = Conv1D(256, 7, padding='same', strides=2, activation='relu', name='con2')(inputs)
    cnn2 = MaxPooling1D(pool_size=4, name='mp2')(cnn2)
    cnn3 = Conv1D(256, 9, padding='same', strides=2, activation='relu', name='con3')(inputs)
    cnn3 = MaxPooling1D(pool_size=2, name='mp3')(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = Flatten(name='flat')(cnn)
    #加上dropout防止过拟合
    drop = Dropout(0.5, name='drop')(flat)
    main_output = Dense(n_outputs, activation='sigmoid', name='output')(drop)
    model = Model(inputs=inputs, outputs=main_output)
    return model

def LSTM_model(n_timesteps, n_features, n_outputs=1, n_gru_units=128, n_dense_units=512):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_gru_units, return_sequences=True, name='lstm1'), input_shape=(n_timesteps, n_features)))
    model.add(Bidirectional(LSTM(n_gru_units, return_sequences=True, name='lstm2')))
    model.add(GlobalAveragePooling1D(name='gap'))
    model.add(BatchNormalization(name='bn'))
    model.add(Dense(n_dense_units, activation='relu', name='fc'))
    model.add(Dropout(0.5, name='dropout'))
    model.add(Dense(n_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(0.05), name='output'))
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['mse'])
    return model

def GRU_model(n_timesteps, n_features, n_outputs=1, n_gru_units=128, n_dense_units=512):
    model = Sequential()
    model.add(Bidirectional(GRU(n_gru_units, return_sequences=True, name='gru1'), input_shape=(n_timesteps, n_features)))
    model.add(Bidirectional(GRU(n_gru_units, return_sequences=True), name='gru2'))
    model.add(GlobalAveragePooling1D(name='gap'))
    model.add(BatchNormalization(name='bn'))
    model.add(Dense(n_dense_units, activation='relu', name='fc'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(0.05), name='output'))
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['mse'])
    return model