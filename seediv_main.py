'''
the data comes from eeg and eye movement .. for eeg there are features extracted, they are PSD features on 5 frequency bands average psd
and there were 62 eeg channels so the dims of features are 62x5 ... the trials are 24 trials and they are videos of different lengths
each trial is segmented into 4 sec. segments with no overlap and there were 3 sessions with different label sequence .. there are 4 classes..
for eye mov. the features are 31 features extracted at each time window (segment) and those are used in classification .. there are 15 ppnts
in this script, we tried evaluation of the models based on within session classification .. or aggregating 3 sessions then classification or .. aggregating 
all ppnts and then classification .. and accord. to paper of BiHDM they use the trials at the end of each session for testing this is good because we shouldn't 
do cross validation on segments because there will be dependency between time points and that would inflate acc. ... also tried leave one ppnt out. in the other script
for the models there are some approaches like using dense layers for eeg and using dense for mov. then taking the activations from middle layers as features
to a third dense model .. 
we tried xgboost on the features reshaped as vector .. tried to use pca and some pre-processing ... tried to use xgboost instead of the third dense ...
then without reshaping so with time dim. to test other models because with this we are putting the segments next to each other we had very few trials .. 
and tried bilstm and tranformer and lengths were padded to match the longest video segments .. 
then tried tranformer again but with the dimension of 62x5 that 5 is 20 because we aggregated features for psd and de for different averaging so that is 62x20
and applying tranf. on that would have attention on different bands that is spectral att. and tried on 20x62 and that would be spatial on channels
in the other script tried 3models, tranf. for eeg and we take activations at the middle then dense for mov. and dense to aggregate 
'''
from transformer_functions import extract_features
from keras.layers import Bidirectional, LSTM
from keras.regularizers import l1, l2
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pandas as pd
from scipy.stats import spearmanr
import pickle
from keras.models import Model
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense, GlobalMaxPooling1D
from keras.models import Sequential
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import hdf5storage
import numpy as np
import tensorflow_addons as tfa
import keras
from transformer_functions import build_model
from mne import create_info
from mne.io import RawArray
from keras.layers import LeakyReLU
import scipy.io
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import mne
from fieldtrip2mne import read_epoched
from keras.optimizers import Adam
from sklearn import preprocessing, model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM
import mat73
from xgboost import XGBClassifier
from mne.time_frequency import tfr_morlet
import matplotlib
import random
import os
from mne.time_frequency import tfr_array_morlet
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.metrics import pairwise_distances
import tensorflow as tf
from keras import backend as K
# matplotlib.use('Agg')

print(tf.config.list_physical_devices('GPU'))


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def nearest(vec, number):
    return np.argmin(abs(vec - number))


def fisher_z(correlations):
    return 0.5 * np.log((1 + correlations) / (1 - correlations))


def classifyXGboost_CV(data, labels):
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True)
    scores = []
    # power = np.squeeze(np.mean(power, axis=2))
    X = data.reshape(data.shape[0], -1)
    for train, val in tqdm(kfold.split(X, labels)):
        # clf = RandomForestClassifier(n_estimators=1000)
        # clf = LinearSVC()
        # scaler = StandardScaler()
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
        clf.fit(X[train], labels[train])
        score = clf.score(X[val], labels[val])
        print(f"Accuracy: {score}")
        scores.append(score)  # append accuracy

    print(scores)
    print(np.mean(scores))


def baseline_correc(data):
    # takes data and correct with data from the first trial .. so it would be difference from first 4sec.
    baseline = data[0, :, :]
    for i in range(data.shape[0]):
        data[i, :, :] = (data[i, :, :] - baseline)
    return data[1:, :, :]


def pad_trls(all_dat, session, trials_co, max_trl_dur, psd):
    # takes 3d and would append windows in the first dim to match that of the max video length and then add dim of trials at the start
    prev_len = 0
    if psd == 1:  # for psd de features
        all_dat_pad = []
        for i in range(0, trials_co.shape[1]):
            if i == 0:
                dat = all_dat[0:int(trials_co[session, i]), :, :]  # because inside ppnt session then 3d
                prev_len += dat.shape[0]
            else:
                dat = all_dat[prev_len: prev_len+int(trials_co[session, i]), :, :]
                prev_len += dat.shape[0]
            num_rows_to_pad = max_trl_dur - dat.shape[0]
            all_dat_pad.append(np.pad(dat, ((0, num_rows_to_pad), (0, 0), (0, 0)), mode='constant')[np.newaxis, :])
    else:  # for mov data because dims are different
        all_dat_pad = []
        for i in range(0, trials_co.shape[1]):
            if i == 0:
                dat = all_dat[0:int(trials_co[session, i]), :]  # because inside ppnt session then 3d
                prev_len += dat.shape[0]
            else:
                dat = all_dat[prev_len: prev_len+int(trials_co[session, i]), :]
                prev_len += dat.shape[0]
            num_rows_to_pad = max_trl_dur - dat.shape[0]
            all_dat_pad.append(np.pad(dat, ((0, num_rows_to_pad), (0, 0)), mode='constant')[np.newaxis, :])
    return all_dat_pad  # trials time(windows) channels bands


def masked_zscore(arr, mask_value=0):
    # Create a masked array where mask_value is masked
    ma = np.ma.masked_equal(arr, mask_value)

    # Calculate mean and standard deviation considering only non-masked values
    mean = np.ma.mean(ma, axis=0)
    std = np.ma.std(ma, axis=0)

    # Subtract the mean and divide by std only where mask is False (i.e., not equal to mask_value)
    arr = np.where(arr != mask_value, (arr - mean) / std, arr)

    return arr


def train_multimodal_dense(Xtrain, Xval, no_features, epochs_count, labels_val, labels_train, reg_weight, Xtest, labels_test):
    # fully connected
    n_classes = 4
    learning_rate = 0.003
    # Convert labels to categorical
    labels_train = to_categorical(labels_train)
    labels_val = to_categorical(labels_val)
    labels_test = to_categorical(labels_test)

    Xtrain1 = Xtrain[:, 0:no_features]
    Xval1 = Xval[:, 0:no_features]
    Xtest1 = Xtest[:, 0:no_features]
    Xtrain2 = Xtrain[:, no_features:]
    Xval2 = Xval[:, no_features:]
    Xtest2 = Xtest[:, no_features:]

    # PCA
    # pca = PCA(n_components=100)
    # Xtrain1 = pca.fit_transform(Xtrain1)
    # Xval1 = pca.transform(Xval1)
    # Xtest1 = pca.transform(Xtest1)

    # pca = PCA(n_components=15)
    # Xtrain2 = pca.fit_transform(Xtrain2)
    # Xval2 = pca.transform(Xval2)
    # Xtest2 = pca.transform(Xtest2)

    # psd and de model
    model = Sequential()
    # model.add(Dense(1000, activation='relu', kernel_regularizer=l2(reg_weight)))
    # model.add(Dense(500, activation='relu', kernel_regularizer=l2(reg_weight)))
    # model.add(Dense(413, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dense(250, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dense(125, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # mu = np.mean(Xtrain, axis=0)
    # sigma = np.std(Xtrain, axis=0)
    model.fit(Xtrain1, labels_train, epochs=epochs_count, verbose=1, validation_data=(Xval1, labels_val))
    model_psd_de = Model(inputs=model.input, outputs=model.layers[-5].output)
    activations_psd_de = model_psd_de.predict(Xtrain1)
    activations_psd_de_Xval = model_psd_de.predict(Xval1)
    activations_psd_de_Xtest = model_psd_de.predict(Xtest1)

    activations_psd_de = stats.zscore(activations_psd_de, axis=0)
    activations_psd_de_Xval = stats.zscore(activations_psd_de_Xval, axis=0)
    activations_psd_de_Xtest = stats.zscore(activations_psd_de_Xtest, axis=0)
    # removing features that become zero or nan
    id = ~((np.mean(activations_psd_de, axis=0) == 0) | (np.isnan(np.mean(activations_psd_de, axis=0))))
    id2 = ~((np.mean(activations_psd_de_Xval, axis=0) == 0) | (np.isnan(np.mean(activations_psd_de_Xval, axis=0))))
    id3 = ~((np.mean(activations_psd_de_Xtest, axis=0) == 0) | (np.isnan(np.mean(activations_psd_de_Xtest, axis=0))))
    # if (not np.array_equal(id, id2)) | (not np.array_equal(id, id3)):
    #     id4 = id & id2 & id3
    #     raise Exception("activations gave zeros in different locations for train and validation!!")
    id = id & id2 & id3  # so that it would take the smaller number of features that are non-zeros
    activations_psd_de = activations_psd_de[:, id]
    activations_psd_de_Xval = activations_psd_de_Xval[:, id]
    activations_psd_de_Xtest = activations_psd_de_Xtest[:, id]

    # mov model
    model = Sequential()  # because features are 31 so we would have less nodes
    # model.add(Dense(1000, activation='relu', kernel_regularizer=l2(reg_weight)))  # L2 regularization
    # model.add(Dense(800, activation='relu', kernel_regularizer=l2(reg_weight)))  # L2 regularization
    # model.add(Dense(500, activation='relu', kernel_regularizer=l2(reg_weight)))  # L2 regularization
    # model.add(Dense(150, activation='relu', kernel_regularizer=l2(reg_weight)))  # L2 regularization
    # # model.add(Dropout(0.5))  # Dropout for regularization
    # model.add(Dense(100, activation='relu', kernel_regularizer=l2(reg_weight)))
    # # model.add(Dropout(0.5))
    # model.add(Dense(40, activation='relu', kernel_regularizer=l2(reg_weight)))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # mu = np.mean(Xtrain, axis=0)
    # sigma = np.std(Xtrain, axis=0)
    model.fit(Xtrain2, labels_train, epochs=epochs_count, verbose=1, validation_data=(Xval2, labels_val))
    model_mov = Model(inputs=model.input, outputs=model.layers[-3].output)
    activations_mov = model_mov.predict(Xtrain2)
    activations_mov_Xval = model_mov.predict(Xval2)
    activations_mov_Xtest = model_mov.predict(Xtest2)

    activations_mov = stats.zscore(activations_mov, axis=0)
    activations_mov_Xval = stats.zscore(activations_mov_Xval, axis=0)
    activations_mov_Xtest = stats.zscore(activations_mov_Xtest, axis=0)

    # removing features that become zero or nan
    id = ~((np.mean(activations_mov, axis=0) == 0) | (np.isnan(np.mean(activations_mov, axis=0))))
    id2 = ~((np.mean(activations_mov_Xval, axis=0) == 0) | (np.isnan(np.mean(activations_mov_Xval, axis=0))))
    id3 = ~((np.mean(activations_mov_Xtest, axis=0) == 0) | (np.isnan(np.mean(activations_mov_Xtest, axis=0))))
    # if (not np.array_equal(id, id2)) | (not np.array_equal(id, id3)):
    # raise Exception("activations gave zeros in different locations for train and validation!!")
    id = id & id2 & id3  # so that it would take the smaller number of features that are non-zeros
    activations_mov = activations_mov[:, id]
    activations_mov_Xval = activations_mov_Xval[:, id]
    activations_mov_Xtest = activations_mov_Xtest[:, id]
    # trying with mov features as is
    # activations_mov = Xtrain[:, no_features:]
    # activations_mov_Xval = Xval[:, no_features:]
    # activations_mov_Xtest = Xtest[:, no_features:]

    # aggregating model
    trn = np.concatenate((activations_psd_de, activations_mov), axis=1)
    val = np.concatenate((activations_psd_de_Xval, activations_mov_Xval), axis=1)
    tst = np.concatenate((activations_psd_de_Xtest, activations_mov_Xtest), axis=1)

    # trn = stats.zscore(trn, axis=0)  # each was zscored so don't have to do this again
    # val = stats.zscore(val, axis=0)
    # tst = stats.zscore(tst, axis=0)

    model = Sequential()
    # model.add(Dense(15, activation='relu', kernel_regularizer=l2(reg_weight)))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dense(5, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # mu = np.mean(Xtrain, axis=0)
    # sigma = np.std(Xtrain, axis=0)
    model.fit(trn, labels_train, epochs=epochs_count, verbose=1, validation_data=(val, labels_val))

    y_probs = model.predict(tst)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(labels_test, axis=1)
    score = accuracy_score(y_true, y_pred)

    # clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
    # # clf = RandomForestClassifier(n_estimators=1000)
    # # clf = LinearSVC()
    # # clf.fit(trn, trn_lbls)
    # # score = clf.score(tst, tst_lbls)
    # print(f"Accuracy: {score}")
    # scores.append(score)  # append accuracy

    return score


def train_multimodal_transformer(Xtrain, Xval, no_features, epochs_count, labels_val, labels_train, reg_weight, Xtest, labels_test, axtime, dims_of_reshape):
    # fully connected
    n_classes = 4
    learning_rate = 0.003
    # Convert labels to categorical
    labels_train = to_categorical(labels_train)
    labels_val = to_categorical(labels_val)
    labels_test = to_categorical(labels_test)

    if axtime == 1:
        Xtrain1 = Xtrain[:, :, 0:no_features]
        Xval1 = Xval[:, :, 0:no_features]
        Xtest1 = Xtest[:, :, 0:no_features]
        Xtrain2 = Xtrain[:, :, no_features:]
        Xval2 = Xval[:, :, no_features:]
        Xtest2 = Xtest[:, :, no_features:]
    else:
        Xtrain1 = Xtrain[:, 0:no_features]
        Xval1 = Xval[:, 0:no_features]
        Xtest1 = Xtest[:, 0:no_features]
        Xtrain2 = Xtrain[:, no_features:]
        Xval2 = Xval[:, no_features:]
        Xtest2 = Xtest[:, no_features:]

    # PCA
    # pca = PCA(n_components=100)
    # Xtrain1 = pca.fit_transform(Xtrain1)
    # Xval1 = pca.transform(Xval1)
    # Xtest1 = pca.transform(Xtest1)

    # pca = PCA(n_components=15)
    # Xtrain2 = pca.fit_transform(Xtrain2)
    # Xval2 = pca.transform(Xval2)
    # Xtest2 = pca.transform(Xtest2)

    # tranformer for psd de and then another for the mov and then a dense
    # psd and de model
    # transformer
    # Xtrain = np.transpose(Xtrain, (0, 2, 1))
    # Xtest = np.transpose(Xtest, (0, 2, 1))
    if axtime == 0:
        Xtrain1 = np.reshape(Xtrain1, [Xtrain1.shape[0], dims_of_reshape[0], dims_of_reshape[1]])
        Xval1 = np.reshape(Xval1, [Xval1.shape[0],  dims_of_reshape[0], dims_of_reshape[1]])
        Xtest1 = np.reshape(Xtest1, [Xtest1.shape[0], dims_of_reshape[0], dims_of_reshape[1]])

        # spatial att. not spectral
        Xtrain1 = np.transpose(Xtrain1, [0, 2, 1])
        Xval1 = np.transpose(Xval1, [0, 2, 1])
        Xtest1 = np.transpose(Xtest1, [0, 2, 1])

        s = 1
    input_shape = Xtrain1.shape[1:]
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[150, 100, 40, 10],  # these are after the multihead attention so for classification so would add these dense layers
        n_classes=n_classes,
        mlp_dropout=0.4,
        dropout=0.25,
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.summary()
    # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)] # if we want to stop after certain epochs if there is no improvement
    model.fit(
        Xtrain1,
        labels_train,
        validation_split=0.1,  # because we have validation externally
        epochs=50,
        # batch_size=1,
        # validation_data=(Xval1, labels_val)
        # callbacks=callbacks,
    )
    # extract_features from last dense in mlp_units
    model_psd_de = extract_features(model)
    activations_psd_de = model_psd_de.predict(Xtrain1)
    activations_psd_de_Xval = model_psd_de.predict(Xval1)
    activations_psd_de_Xtest = model_psd_de.predict(Xtest1)

    activations_psd_de = stats.zscore(activations_psd_de, axis=0)
    activations_psd_de_Xval = stats.zscore(activations_psd_de_Xval, axis=0)
    activations_psd_de_Xtest = stats.zscore(activations_psd_de_Xtest, axis=0)
    # removing features that become zero or nan
    id = ~((np.mean(activations_psd_de, axis=0) == 0) | (np.isnan(np.mean(activations_psd_de, axis=0))))
    id2 = ~((np.mean(activations_psd_de_Xval, axis=0) == 0) | (np.isnan(np.mean(activations_psd_de_Xval, axis=0))))
    id3 = ~((np.mean(activations_psd_de_Xtest, axis=0) == 0) | (np.isnan(np.mean(activations_psd_de_Xtest, axis=0))))
    # if (not np.array_equal(id, id2)) | (not np.array_equal(id, id3)):
    #     id4 = id & id2 & id3
    #     raise Exception("activations gave zeros in different locations for train and validation!!")
    id = id & id2 & id3  # so that it would take the smaller number of features that are non-zeros
    activations_psd_de = activations_psd_de[:, id]
    activations_psd_de_Xval = activations_psd_de_Xval[:, id]
    activations_psd_de_Xtest = activations_psd_de_Xtest[:, id]

    if axtime == 1:  # because if axtime then we would try to use tranf. on the time dim but if not then that is ch bands and mov don't include channels
        # so we use dense in else ..
        # mov model
        input_shape = Xtrain2.shape[1:]
        model = build_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[20, 10],  # these are after the multihead attention so for classification so would add these dense layers
            n_classes=n_classes,
            mlp_dropout=0.4,
            dropout=0.25,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        model.summary()
        # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)] # if we want to stop after certain epochs if there is no improvement
        model.fit(
            Xtrain2,
            labels_train,
            validation_split=0.1,  # because we have validation externally
            epochs=50,
            # batch_size=1,
            # validation_data=(Xval1, labels_val)
            # callbacks=callbacks,
        )
        # extract_features from last dense in mlp_units
        model_mov = extract_features(model)
        activations_mov = model_mov.predict(Xtrain2)
        activations_mov_Xval = model_mov.predict(Xval2)
        activations_mov_Xtest = model_mov.predict(Xtest2)

        activations_mov = stats.zscore(activations_mov, axis=0)
        activations_mov_Xval = stats.zscore(activations_mov_Xval, axis=0)
        activations_mov_Xtest = stats.zscore(activations_mov_Xtest, axis=0)

        # removing features that become zero or nan
        id = ~((np.mean(activations_mov, axis=0) == 0) | (np.isnan(np.mean(activations_mov, axis=0))))
        id2 = ~((np.mean(activations_mov_Xval, axis=0) == 0) | (np.isnan(np.mean(activations_mov_Xval, axis=0))))
        id3 = ~((np.mean(activations_mov_Xtest, axis=0) == 0) | (np.isnan(np.mean(activations_mov_Xtest, axis=0))))
        # if (not np.array_equal(id, id2)) | (not np.array_equal(id, id3)):
        # raise Exception("activations gave zeros in different locations for train and validation!!")
        id = id & id2 & id3  # so that it would take the smaller number of features that are non-zeros
        activations_mov = activations_mov[:, id]
        activations_mov_Xval = activations_mov_Xval[:, id]
        activations_mov_Xtest = activations_mov_Xtest[:, id]
    else:
        # mov model
        model = Sequential()  # because features are 31 so we would have less nodes
        # model.add(Dense(100, activation='relu', kernel_regularizer=l2(reg_weight)))
        # # model.add(Dropout(0.5))
        # model.add(Dense(40, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        # mu = np.mean(Xtrain, axis=0)
        # sigma = np.std(Xtrain, axis=0)
        model.fit(Xtrain2, labels_train, epochs=epochs_count, verbose=1, validation_data=(Xval2, labels_val))
        model_mov = Model(inputs=model.input, outputs=model.layers[-3].output)
        activations_mov = model_mov.predict(Xtrain2)
        activations_mov_Xval = model_mov.predict(Xval2)
        activations_mov_Xtest = model_mov.predict(Xtest2)

        activations_mov = stats.zscore(activations_mov, axis=0)
        activations_mov_Xval = stats.zscore(activations_mov_Xval, axis=0)
        activations_mov_Xtest = stats.zscore(activations_mov_Xtest, axis=0)

        # removing features that become zero or nan
        id = ~((np.mean(activations_mov, axis=0) == 0) | (np.isnan(np.mean(activations_mov, axis=0))))
        id2 = ~((np.mean(activations_mov_Xval, axis=0) == 0) | (np.isnan(np.mean(activations_mov_Xval, axis=0))))
        id3 = ~((np.mean(activations_mov_Xtest, axis=0) == 0) | (np.isnan(np.mean(activations_mov_Xtest, axis=0))))
        # if (not np.array_equal(id, id2)) | (not np.array_equal(id, id3)):
        # raise Exception("activations gave zeros in different locations for train and validation!!")
        id = id & id2 & id3  # so that it would take the smaller number of features that are non-zeros
        activations_mov = activations_mov[:, id]
        activations_mov_Xval = activations_mov_Xval[:, id]
        activations_mov_Xtest = activations_mov_Xtest[:, id]
    # trying with mov features as is
    # activations_mov = Xtrain[:, no_features:]
    # activations_mov_Xval = Xval[:, no_features:]
    # activations_mov_Xtest = Xtest[:, no_features:]

    # aggregating model
    trn = np.concatenate((activations_psd_de, activations_mov), axis=1)
    val = np.concatenate((activations_psd_de_Xval, activations_mov_Xval), axis=1)
    tst = np.concatenate((activations_psd_de_Xtest, activations_mov_Xtest), axis=1)

    # trn = stats.zscore(trn, axis=0)  # each was zscored so don't have to do this again
    # val = stats.zscore(val, axis=0)
    # tst = stats.zscore(tst, axis=0)

    model = Sequential()
    # model.add(Dense(15, activation='relu', kernel_regularizer=l2(reg_weight)))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dense(5, activation='relu', kernel_regularizer=l2(reg_weight)))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # mu = np.mean(Xtrain, axis=0)
    # sigma = np.std(Xtrain, axis=0)
    model.fit(trn, labels_train, epochs=epochs_count, verbose=1, validation_data=(val, labels_val))

    y_probs = model.predict(tst)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(labels_test, axis=1)
    score = accuracy_score(y_true, y_pred)

    # clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
    # # clf = RandomForestClassifier(n_estimators=1000)
    # # clf = LinearSVC()
    # # clf.fit(trn, trn_lbls)
    # # score = clf.score(tst, tst_lbls)
    # print(f"Accuracy: {score}")
    # scores.append(score)  # append accuracy

    return score


# %% params
seed_everything(42)

# path = 'D:/Literature papers/Changing pattern speed/AudStream_RawData/AC1/A.cnt'
# raw = mne.io.read_raw_cnt(path)


# events, event_dict = mne.events_from_annotations(raw)
# raw.plot(block=True)
# fig = mne.viz.plot_events(
#     events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_dict
# )
# try to load the data from mat
# start of the deviant in cond. and this is in ms
# cond d i n are different because the shift is after the tone so not sure how we would segment them would try something now after the tone but
# guess this would need to be changed
permutations = 1000
shuffle = 0
dense = 1  # whether to call 3 dense models or tranformer
axtime = 0  # whether to call tranformer model on the time dim or to call it on ch dim. .. if we want ch feat we put axtime = 0 and dense = 1 so that
# we get trials from windows


sim_val = []
files = []
all_trials = []
all_labels = []
preprocess = 0

window_seconds = 5

session_labels = []
session_labels.append([1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3])  # 0 neutral, 1 sad, 2 fear, 3 happy
session_labels.append([2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1])
session_labels.append([1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0])


training_pos = []  # training would be 1 and testing 0
training_pos.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
training_pos.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
training_pos.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])


ppnt_lbls = []
ppnt_dat = []
ppnt_dat_mov = []
trials_co = np.empty((3, 24))
for session_no in range(0, 3):  # sessions
    pathm = './seed/SEED_IV/eeg_feature_smooth/' + str(session_no+1) + '/'
    for file in tqdm(os.listdir(pathm)):  # ppnts
        data = scipy.io.loadmat(pathm + file)
        dat = []
        lbls = []
        dat_mov = []
        for trl in range(0, 24):  # trials
            temp = baseline_correc(np.transpose(data['psd_movingAve' + str(trl+1)], (1, 0, 2)))  # to could put the first dim as 1 so that it would still
            trials_co[session_no, trl] = temp.shape[0]
            # be trials and then the second dim would be windows so that it would be time points .. so 1_trial_ch_band instead of trial_ch_band
            temp2 = baseline_correc(np.transpose(data['de_movingAve' + str(trl+1)], (1, 0, 2)))  # to trial_ch_band ... adding de features
            temp3 = baseline_correc(np.transpose(data['de_LDS' + str(trl+1)], (1, 0, 2)))  # to trial_ch_band ... adding features
            temp4 = baseline_correc(np.transpose(data['psd_LDS' + str(trl+1)], (1, 0, 2)))  # to trial_ch_band ... adding features
            # choosing some channels that may have the features ..
            channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
                             'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                             'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
                             'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
                             ]
            # channels_of_analysis = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
            #                         'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            #                         'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
            #                         'P2', 'P4', 'P6', 'P8'
            #                         # 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2' .. so would remove 12 from the last channels
            #                         ]
            # temp = temp[np.newaxis, :]
            # temp2 = temp2[np.newaxis, :]
            # temp3 = temp3[np.newaxis, :]
            # temp4 = temp4[np.newaxis, :]
            temp = np.concatenate((temp, temp2, temp3, temp4), axis=2)
            # temp = temp2
            # temp = temp[:, :-12, :]
            # temp = np.mean(temp, axis=1)
            # loading eye mov features..
            data_mov = scipy.io.loadmat('./seed/SEED_IV/eye_feature_smooth/' + str(session_no+1) + '/' + file)
            temp_move = np.transpose(data_mov['eye_' + str(trl+1)])
            # temp_move = temp_move[np.newaxis, :]
            if np.isnan(np.mean(temp_move)):
                s = 1
            dat_mov.append(temp_move[1:, :])  # considering the first trial as baseline ..
            temp_labels = np.repeat(session_labels[session_no][trl], temp.shape[0])
            dat.append(temp)
            lbls.append(temp_labels)
        ppnt_dat_mov.append(np.concatenate(dat_mov, axis=0))
        ppnt_dat.append(np.concatenate(dat, axis=0))
        ppnt_lbls.append(np.concatenate(lbls, axis=0))

# ppnt_dat and ppnt_lbls .. should have 15 ppnts data and this is repeated 3 times one for every session ..
# will leave all channels because all are eeg

all_dat = []  # this would loop and add ppnt data for every session so would have a list of no. ppnts and inside it the data for each session
all_lbls = []
all_mov = []
for i in range(0, 15):  # ppnts
    temp = []
    temp_lbls = []
    temp_mov = []
    for ppnt in range(0, len(ppnt_dat), 15):
        print(ppnt+i)
        temp.append(ppnt_dat[ppnt+i])
        temp_mov.append(ppnt_dat_mov[ppnt+i])
        temp_lbls.append(ppnt_lbls[ppnt+i])
    all_mov.append(temp_mov)
    all_dat.append(temp)
    all_lbls.append(temp_lbls)


# classification
# scores = []
# for i in range(0, 15):  # ppnts
#     # aggregating all sessions
#     # data = np.concatenate(all_dat[i], axis=0)  # ppnt_session .. aggregating sessions for the same ppnt
#     # labels = np.concatenate(all_lbls[i], axis=0)  # ppnt_session
#     # classifyXGboost_CV(data, labels)

#     # two sessions for training and the other for testing
#     data = np.concatenate((all_dat[i][0], all_dat[i][1]), axis=0)
#     labels = np.concatenate((all_lbls[i][0], all_lbls[i][1]), axis=0)

#     data_test = all_dat[i][2]
#     labels_test = all_lbls[i][2]

#     Xtrain = data.reshape(data.shape[0], -1)
#     Xtest = data_test.reshape(data_test.shape[0], -1)

#     clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
#     clf.fit(Xtrain, labels)
#     score = clf.score(Xtest, labels_test)
#     print(f"Accuracy: {score}")
#     scores.append(score)  # append accuracy

# s = 1
# print(scores)
# print(np.mean(scores))

# leave one ppnt out
ppnts_no = 15
scores = []
max_trl_dur = int(np.max(trials_co))

ids = [_ for _ in range(0, ppnts_no)]

for i in range(0, ppnts_no):  # ppnts
    temp = []
    temp_labels = []
    temp_test = []
    temp_mov = []
    temp_labels_test = []
    temp_test_mov = []
    keep_id = [ids[_] for _ in ids if _ != i]
    # for k in keep_id:

    trn = []
    tst = []
    trn_lbls = []
    tst_lbls = []
    for sess in range(0, 3):  # session

        # psd de
        # with reshape it should be same channel and different bands next to each other
        dat_pad = pad_trls(all_dat[i][sess], sess, trials_co, max_trl_dur, 1)
        dat_pad = np.concatenate(dat_pad, axis=0)
        # reshaping so that features would be in 3rd dim and would be same channel different bands next to each other so should be trial time features
        # temp.append(np.reshape(dat_pad, (dat_pad.shape[0], dat_pad.shape[1], dat_pad.shape[2]*dat_pad.shape[3])))
        psd_dat = np.reshape(dat_pad, (dat_pad.shape[0], dat_pad.shape[1], dat_pad.shape[2]*dat_pad.shape[3]))
        dims_of_reshape = [dat_pad.shape[2], dat_pad.shape[3]]

        # mov
        dat_pad = pad_trls(all_mov[i][sess], sess, trials_co, max_trl_dur, 0)
        dat_pad = np.concatenate(dat_pad, axis=0)
        # temp_mov.append(dat_pad)
        mov_dat = dat_pad
        all_feats = np.concatenate((psd_dat, mov_dat), axis=2)

        # labels
        # temp_labels.append(session_labels[sess])
        label = np.array(session_labels[sess])
        # split training and testing
        trn.append(all_feats[[_ == 1 for _ in training_pos[sess]], :, :])
        tst.append(all_feats[[_ == 0 for _ in training_pos[sess]], :, :])
        trn_lbls.append(label[[_ == 1 for _ in training_pos[sess]]])
        tst_lbls.append(label[[_ == 0 for _ in training_pos[sess]]])

    # %% machine learning .. trials features
    trn = np.concatenate(trn, axis=0)
    tst = np.concatenate(tst, axis=0)
    trn_lbls = np.concatenate(trn_lbls, axis=0)
    tst_lbls = np.concatenate(tst_lbls, axis=0)
    # trial with nan
    if np.isnan(np.mean(all_feats)):
        trial_notnan_id = np.where(~np.isnan(np.mean(trn, axis=(1, 2))))[0]
        trn = trn[trial_notnan_id, :, :]
        trn_lbls = trn_lbls[trial_notnan_id]

        trial_notnan_id = np.where(~np.isnan(np.mean(tst, axis=(1, 2))))[0]
        tst = tst[trial_notnan_id, :, :]
        tst_lbls = tst_lbls[trial_notnan_id]

    if dense == 1:
        # data should be trials timewindows features .. if we want to use xgboost and so on we would reshape .. but !!
        # this will make problems because we padded features with zeros .. so we remove the trials that would be zero
        trn_lbls = np.tile(trn_lbls, trn.shape[1])
        trn = np.transpose(trn, [2, 1, 0])  # put trials at the end so that it would be columns in the 2d timewindows trials
        trn = np.reshape(trn, (trn.shape[0], -1)).T

        tst_lbls = np.tile(tst_lbls, tst.shape[1])
        tst = np.transpose(tst, [2, 1, 0])  # put trials at the end so that it would be columns in the 2d timewindows trials
        tst = np.reshape(tst, (tst.shape[0], -1)).T

        # removing trials that would be zero because those came from padding
        keep_trn = np.mean(trn, axis=1) != 0
        keep_tst = np.mean(tst, axis=1) != 0
        trn = trn[keep_trn, :]
        trn_lbls = trn_lbls[keep_trn]
        tst = tst[keep_tst, :]
        tst_lbls = tst_lbls[keep_tst]

    trn, val, trn_lbls, val_lbls = train_test_split(trn, trn_lbls, test_size=0.1)

    trn = stats.zscore(trn, axis=0)
    val = stats.zscore(val, axis=0)
    tst = stats.zscore(tst, axis=0)

    # clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)

    no_features = psd_dat.shape[-1]
    epochs_count = 50

    # score = train_multimodal_dense(trn, val, no_features, epochs_count, val_lbls, trn_lbls, 0.001, tst, tst_lbls)
    score = train_multimodal_transformer(trn, val, no_features, epochs_count, val_lbls, trn_lbls, 0.001, tst, tst_lbls, axtime, dims_of_reshape)

    # clf = RandomForestClassifier(n_estimators=1000)
    # clf = LinearSVC()
    # clf.fit(trn, trn_lbls)
    # score = clf.score(tst, tst_lbls)
    print(f"Accuracy: {score}")
    scores.append(score)  # append accuracy

s = 1
s = 1
print(scores)
print(np.mean(scores))
print(np.std(scores))

# # %% trying on the timeseries ... trial time features
# trn = np.concatenate(trn, axis=0)
# tst = np.concatenate(tst, axis=0)
# trn_lbls = np.concatenate(trn_lbls, axis=0)
# tst_lbls = np.concatenate(tst_lbls, axis=0)

# # trial with nan
# if np.isnan(np.mean(all_feats)):
#     trial_notnan_id = np.where(~np.isnan(np.mean(trn, axis=(1, 2))))[0]
#     trn = trn[trial_notnan_id, :, :]
#     trn_lbls = trn_lbls[trial_notnan_id]

#     trial_notnan_id = np.where(~np.isnan(np.mean(tst, axis=(1, 2))))[0]
#     tst = tst[trial_notnan_id, :, :]
#     tst_lbls = tst_lbls[trial_notnan_id]

# n_classes = len(np.unique(trn_lbls))
# trn_lbls = to_categorical(trn_lbls)
# tst_lbls = to_categorical(tst_lbls)
# epochs_count = 50
# learning_rate = 0.001

# # trn = masked_zscore(trn)
# # tst = masked_zscore(tst)

# # bilstm
# model = Sequential()
# # Add a Bidirectional LSTM layer instead of Dense layers
# model.add(Bidirectional(LSTM(200, return_sequences=True), input_shape=(trn.shape[1], trn.shape[2])))  # timesteps features
# model.add(Bidirectional(LSTM(150, return_sequences=True), input_shape=(trn.shape[1], 200)))
# model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(trn.shape[1], 150)))
# model.add(Bidirectional(LSTM(50, input_shape=(trn.shape[1], 100))))
# model.add(Dropout(0.5))
# model.add(Dense(n_classes, activation='softmax'))
# model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(trn, trn_lbls, epochs=epochs_count, verbose=1)

# y_probs = model.predict(tst)
# y_pred = np.argmax(y_probs, axis=1)
# y_true = np.argmax(tst_lbls, axis=1)
# score = accuracy_score(y_true, y_pred)
# print(f"Accuracy: {score}")
# scores.append(score)

# # # transformer
# # # Xtrain = np.transpose(Xtrain, (0, 2, 1))
# # # Xtest = np.transpose(Xtest, (0, 2, 1))
# # input_shape = trn.shape[1:]
# # model = build_model(
# #     input_shape,
# #     head_size=256,
# #     num_heads=4,
# #     ff_dim=4,
# #     num_transformer_blocks=4,
# #     mlp_units=[150, 100, 40, 10],  # these are after the multihead attention so for classification so would add these dense layers
# #     n_classes=n_classes,
# #     mlp_dropout=0.4,
# #     dropout=0.25,
# # )
# # model.compile(
# #     loss="categorical_crossentropy",
# #     optimizer=keras.optimizers.Adam(learning_rate=0.001),
# #     metrics=["accuracy"],
# # )
# # model.summary()
# # # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)] # if we want to stop after certain epochs if there is no improvement
# # model.fit(
# #     trn,
# #     trn_lbls,
# #     # validation_split=0.2, # because we have validation externally
# #     epochs=50,
# #     batch_size=1,
# #     # callbacks=callbacks,
# # )

# # predictions = model.predict(tst)
# # score = accuracy_score(np.argmax(tst_lbls, axis=1), np.argmax(predictions, axis=1))
# # print(f"Accuracy: {score}")
# # scores.append(score)  # append accuracy

# s = 1


s = 1
print(scores)
print(np.mean(scores))
print(np.std(scores))
