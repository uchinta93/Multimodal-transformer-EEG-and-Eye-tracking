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
from keras.layers import Input, Dense, Add, concatenate
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
import keras
from transformer_functions import build_model
from transformer_functions import extract_features
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
import tensorflow_addons as tfa
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
        # for j in range(data.shape[1]):
        # data[i, j, :] = (data[i, j, :] - np.mean(data[i, :, :], axis=0))
    # trying commong average ref.
    # data = data - np.repeat((np.mean(data, axis=1))[:, np.newaxis], data.shape[1], axis=1)
    return data[1:, :, :]


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
        epochs=10,
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
            epochs=10,
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
        model_mov = Model(inputs=model.input, outputs=model.layers[-5].output)
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


def train_multimodal_transformer2(Xtrain, no_features, epochs_count, labels_train, reg_weight, Xtest, labels_test, axtime, dims_of_reshape, data2):
    # one model that takes the features of both and then gives the features of mov. to the tranf. so that it would use both features for optimi.
    # fully connected
    n_classes = 4
    learning_rate = 0.003
    # Convert labels to categorical
    # labels_train = to_categorical(labels_train)
    # labels_val = to_categorical(labels_val)
    # labels_test = to_categorical(labels_test)

    # Xtrain1 = Xtrain[:, 0:no_features]
    Xtest1 = stats.zscore(Xtest[:, 0:no_features], axis=0)
    Xtrain2 = Xtrain[:, no_features:]
    Xtest2 = stats.zscore(Xtest[:, no_features:], axis=0)

    if axtime == 0:
        Xtrain1 = np.reshape(Xtrain, [Xtrain.shape[0], dims_of_reshape[0], dims_of_reshape[1]])
        # Xval1 = np.reshape(Xval1, [Xval1.shape[0],  dims_of_reshape[0], dims_of_reshape[1]])
        Xtest1 = np.reshape(Xtest1, [Xtest1.shape[0], dims_of_reshape[0], dims_of_reshape[1]])

        # spatial att. not spectral
        Xtrain1 = np.transpose(Xtrain1, [0, 2, 1])
        # Xval1 = np.transpose(Xval1, [0, 2, 1])
        Xtest1 = np.transpose(Xtest1, [0, 2, 1])

        s = 1
    input_shape = Xtrain1.shape[1:]
    model = build_model(
        input_shape,
        head_size=10,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[150, 100, 40],  # these are after the multihead attention so for classification so would add these dense layers
        n_classes=n_classes,
        mlp_dropout=0,
        dropout=0,
        data2=data2
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.summary()
    # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)] # if we want to stop after certain epochs if there is no improvement
    model.fit(
        [Xtrain1, data2],
        labels_train,
        # validation_split=0.1,  # because we have validation externally
        epochs=20,
        # batch_size=1,
        # validation_data=(Xval1, labels_val)
        # callbacks=callbacks,
    )

    y_probs = model.predict([Xtest1, Xtest2])
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
dense = 2

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

ppnt_lbls = []
ppnt_dat = []
ppnt_dat_mov = []
for session_no in range(0, 3):  # sessions
    pathm = './seed/SEED_IV/eeg_feature_smooth/' + str(session_no+1) + '/'
    for file in tqdm(os.listdir(pathm)):  # ppnts
        data = scipy.io.loadmat(pathm + file)
        dat = []
        lbls = []
        dat_mov = []
        for trl in range(0, 24):  # trials
            temp = baseline_correc(np.transpose(data['psd_movingAve' + str(trl+1)], (1, 0, 2)))  # to trial_ch_band
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
            temp = np.concatenate((temp, temp2, temp3, temp4), axis=2)
            # temp = temp[:, :-12, :]
            # temp = np.mean(temp, axis=1)
            # loading eye mov features..
            data_mov = scipy.io.loadmat('./seed/SEED_IV/eye_feature_smooth/' + str(session_no+1) + '/' + file)
            temp_move = np.transpose(data_mov['eye_' + str(trl+1)])
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
ids = [_ for _ in range(0, ppnts_no)]
for i in range(0, ppnts_no):  # ppnts
    temp = []
    temp_labels = []
    temp_test = []
    temp_mov = []
    temp_labels_test = []
    temp_test_mov = []
    keep_id = [ids[_] for _ in ids if _ != i]
    for k in keep_id:
        temp.append(stats.zscore(all_dat[k][0], axis=0))  # [k][0]: ppnt session zscore_data = stats.zscore(data, axis=0)
        temp_mov.append(all_mov[k][0])  # [k][0]: but mov featrures
        temp_labels.append(all_lbls[k][0])
        temp.append(stats.zscore(all_dat[k][1], axis=0))
        temp_mov.append(all_mov[k][1])  # [k][0]: but mov featrures
        temp_labels.append(all_lbls[k][1])
        temp.append(stats.zscore(all_dat[k][2], axis=0))
        temp_mov.append(all_mov[k][2])  # [k][0]: but mov featrures
        temp_labels.append(all_lbls[k][2])
    data = np.concatenate(temp, axis=0)  # kept trials
    data_mov = np.concatenate(temp_mov, axis=0)
    labels = np.concatenate(temp_labels, axis=0)
    no_features = data.shape[1] * data.shape[2]
    dims_of_reshape = [data.shape[1], data.shape[2]]
    Xtrain = np.concatenate((data.reshape(data.shape[0], -1), data_mov.reshape(data_mov.shape[0], -1)), axis=1)  # data for transf.
    # out
    temp_test.append(stats.zscore(all_dat[i][0], axis=0))
    temp_test_mov.append(all_mov[i][0])
    temp_labels_test.append(all_lbls[i][0])
    temp_test.append(stats.zscore(all_dat[i][1], axis=0))
    temp_test_mov.append(all_mov[i][1])
    temp_labels_test.append(all_lbls[i][1])
    temp_test.append(stats.zscore(all_dat[i][2], axis=0))
    temp_test_mov.append(all_mov[i][2])
    temp_labels_test.append(all_lbls[i][2])

    data = np.concatenate(temp_test, axis=0)
    data_mov = np.concatenate(temp_test_mov, axis=0)
    labels_test = np.concatenate(temp_labels_test, axis=0)
    Xtest = np.concatenate((data.reshape(data.shape[0], -1), data_mov.reshape(data_mov.shape[0], -1)), axis=1)  # data for transf.

    # pca = PCA(n_components=100)
    # Xtrain = pca.fit_transform(Xtrain)
    # Xtest = pca.transform(Xtest)

    # clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
    # # clf = RandomForestClassifier(n_estimators=1000)
    # # clf = LinearSVC()
    # clf.fit((Xtrain), labels)
    # score = clf.score((Xtest), labels_test)
    # print(f"Accuracy: {score}")
    # scores.append(score)  # append accuracy
    # continue

    #  connected
    # n_classes = len(np.unique(labels))
    # labels = to_categorical(labels)
    # labels_test = to_categorical(labels_test)
    # model = Sequential()
    # model.add(Dense(150, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(40, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(4, activation='softmax'))
    # model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(stats.zscore(Xtrain, axis=0), labels, epochs=20, verbose=1)
    # y_probs = model.predict(stats.zscore(Xtest, axis=0))
    # y_pred = np.argmax(y_probs, axis=1)
    # y_true = np.argmax(labels_test, axis=1)
    # score = accuracy_score(y_true, y_pred)
    # print(f"Accuracy: {score}")
    # scores.append(score)  # append accuracy

    #  connected with regularisation and validation set ..
    # n_classes = len(np.unique(labels))
    # Split your training data into training and validation sets
    # Xtrain, Xval, labels_train, labels_val = train_test_split(Xtrain, labels, test_size=0.2)
    # Convert labels to categorical
    # labels_train = to_categorical(labels_train)
    # labels_val = to_categorical(labels_val)
    # labels_test = to_categorical(labels_test)

    # model = Sequential()
    # model.add(Dense(150, activation='relu', kernel_regularizer=l2(0.01)))  # L2 regularization
    # # model.add(Dropout(0.5))  # Dropout for regularization
    # model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
    # # model.add(Dropout(0.5))
    # model.add(Dense(40, activation='relu', kernel_regularizer=l2(0.01)))
    # # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
    # model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(stats.zscore(Xtrain, axis=0), labels_train, epochs=100, verbose=1, validation_data=(stats.zscore(Xval, axis=0), labels_val))
    # y_probs = model.predict(stats.zscore(Xtest, axis=0))
    # y_pred = np.argmax(y_probs, axis=1)
    # y_true = np.argmax(labels_test, axis=1)
    # score = accuracy_score(y_true, y_pred)
    # print(f"Accuracy: {score}")
    # scores.append(score)

    # models one for psd and de features then one for mov and then one for them together
    epochs_count = 50
    learning_rate = 0.001
    reg_weight = 0.01  # the higher the less the weights

    # removing nan trials ..
    nan_indices = np.where(np.isnan(Xtrain))
    print(f"no. trials to be removed: {len(np.unique(nan_indices[0]))}")
    nan_indices = np.where(np.isnan(Xtest))
    print(f"no. trials to be removed testing: {len(np.unique(nan_indices[0]))}")
    trn_id = ~np.any(np.isnan(Xtrain), axis=1)
    Xtrain = Xtrain[trn_id]
    labels = labels[trn_id]
    tst_id = ~np.any(np.isnan(Xtest), axis=1)
    Xtest = Xtest[tst_id]
    labels_test = labels_test[tst_id]

    n_classes = len(np.unique(labels))
    # Split your training data into training and validation sets
    # Xtrain, Xval, labels_train, labels_val = train_test_split(Xtrain, labels, test_size=0.2)
    # standardising
    # Xtrain = stats.zscore(Xtrain, axis=0)
    # Xval = stats.zscore(Xval, axis=0)
    # Xtest = stats.zscore(Xtest, axis=0)

    if dense == -1:
        # fully connected
        # Convert labels to categorical
        # labels_train = to_categorical(labels_train)
        # labels_val = to_categorical(labels_val)
        # labels_test = to_categorical(labels_test)

        # # psd and de model
        # model = Sequential()
        # model.add(Dense(1000, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dense(500, activation='relu', kernel_regularizer=l2(reg_weight)))
        # # model.add(Dense(413, activation='relu', kernel_regularizer=l2(reg_weight)))
        # # model.add(Dense(250, activation='relu', kernel_regularizer=l2(reg_weight)))
        # # model.add(Dense(125, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dense(60, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dropout(0.5))
        # model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
        # model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        # # mu = np.mean(Xtrain, axis=0)
        # # sigma = np.std(Xtrain, axis=0)
        # model.fit(Xtrain, labels_train, epochs=epochs_count, verbose=1, validation_data=(Xval, labels_val))

        # y_probs = model.predict(Xtest)
        # y_pred = np.argmax(y_probs, axis=1)
        # y_true = np.argmax(labels_test, axis=1)
        # score = accuracy_score(y_true, y_pred)
        # print(f"Accuracy: {score}")
        # scores.append(score)
        #  connected
        # n_classes = len(np.unique(labels))
        labels = to_categorical(labels)
        labels_test = to_categorical(labels_test)
        model = Sequential()
        model.add(Dense(150, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(stats.zscore(Xtrain[:, 0:no_features], axis=0), labels, epochs=20, verbose=1)
        y_probs1 = model.predict(stats.zscore(Xtest[:, 0:no_features], axis=0))
        w1 = history.history['accuracy'][-1]

        model = Sequential()
        model.add(Dense(10, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(stats.zscore(Xtrain[:, no_features:], axis=0), labels, epochs=20, verbose=1)
        y_probs2 = model.predict(stats.zscore(Xtest[:, no_features:], axis=0))
        w2 = history.history['accuracy'][-1]
        # y_probs = (y_probs1 + y_probs2)/2

        w3 = w1+w2
        w1 = w1/w3  # like softm. for weights to have from 0 to 1
        w2 = w2/w3
        y_probs = (w1 * y_probs1 + w2 * y_probs2) / (w1 + w2)  # weighted average because normal average is 1 weights and we divide by 2 the sum of the weights
        y_pred = np.argmax(y_probs, axis=1)
        y_true = np.argmax(labels_test, axis=1)
        score = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {score}")
        scores.append(score)  # append accuracy

    elif dense == -2:
        labels = to_categorical(labels)
        labels_test = to_categorical(labels_test)
        trn = Xtrain[:, no_features:]

        inputA = Input(shape=(no_features,))
        inputB = Input(shape=(trn.shape[-1],))

        x = Dense(150, activation='relu')(inputA)
        x = Dense(100, activation='relu')(x)
        x = Dense(40, activation='relu')(x)

        # Second branch
        y = Dense(trn.shape[-1], activation='relu')(inputB)
        combined = concatenate([x, y])

        z = Dense(10, activation='relu')(combined)
        z = Dense(4, activation='softmax')(z)

        # Our final model will accept inputs from the two branches and output a single value
        model = Model(inputs=[inputA, inputB], outputs=z)

        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([stats.zscore(Xtrain[:, 0:no_features], axis=0), stats.zscore(Xtrain[:, no_features:], axis=0)], labels, epochs=20, verbose=1)
        y_probs = model.predict([stats.zscore(Xtest[:, 0:no_features], axis=0), stats.zscore(Xtest[:, no_features:], axis=0)])

        y_pred = np.argmax(y_probs, axis=1)
        y_true = np.argmax(labels_test, axis=1)
        score = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {score}")
        scores.append(score)  # append accuracy

    elif dense == 1:
        # fully connected
        # Convert labels to categorical
        labels_train = to_categorical(labels_train)
        labels_val = to_categorical(labels_val)
        labels_test = to_categorical(labels_test)

        # psd and de model
        model = Sequential()
        model.add(Dense(1000, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dense(500, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dense(413, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dense(250, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dense(125, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dense(60, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        # mu = np.mean(Xtrain, axis=0)
        # sigma = np.std(Xtrain, axis=0)
        model.fit(Xtrain[:, 0:no_features], labels_train, epochs=epochs_count, verbose=1, validation_data=(Xval[:, 0:no_features], labels_val))
        model_psd_de = Model(inputs=model.input, outputs=model.layers[-3].output)
        activations_psd_de = model_psd_de.predict(Xtrain[:, 0:no_features])
        activations_psd_de_Xval = model_psd_de.predict(Xval[:, 0:no_features])
        activations_psd_de_Xtest = model_psd_de.predict(Xtest[:, 0:no_features])

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
        model.add(Dense(5, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        # mu = np.mean(Xtrain, axis=0)
        # sigma = np.std(Xtrain, axis=0)
        model.fit(Xtrain[:, no_features:], labels_train, epochs=epochs_count, verbose=1, validation_data=(Xval[:, no_features:], labels_val))
        model_mov = Model(inputs=model.input, outputs=model.layers[-3].output)
        activations_mov = model_mov.predict(Xtrain[:, no_features:])
        activations_mov_Xval = model_mov.predict(Xval[:, no_features:])
        activations_mov_Xtest = model_mov.predict(Xtest[:, no_features:])

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
        # trn = stats.zscore(trn, axis=0) # each was zscored so don't have to do this again
        # val = stats.zscore(val, axis=0)
        model = Sequential()
        # model.add(Dense(15, activation='relu', kernel_regularizer=l2(reg_weight)))
        # model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dense(5, activation='relu', kernel_regularizer=l2(reg_weight)))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))  # Use n_classes instead of hardcoding
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        # mu = np.mean(Xtrain, axis=0)
        # sigma = np.std(Xtrain, axis=0)
        model.fit(trn, labels_train, epochs=epochs_count, verbose=1, validation_data=(val, labels_val))

        y_probs = model.predict(tst)
        y_pred = np.argmax(y_probs, axis=1)
        y_true = np.argmax(labels_test, axis=1)
        score = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {score}")
        scores.append(score)
    elif dense == 2:
        # tranf. with 3 models
        axtime = 0
        # score = train_multimodal_dense(trn, val, no_features, epochs_count, val_lbls, trn_lbls, 0.001, tst, tst_lbls)
        labels = to_categorical(labels)
        labels_test = to_categorical(labels_test)

        score = train_multimodal_transformer2(stats.zscore(Xtrain[:, 0:no_features], axis=0), no_features, epochs_count,
                                              labels, 0.001, Xtest, labels_test, axtime, dims_of_reshape, stats.zscore(Xtrain[:, no_features:], axis=0))
        print(f"Accuracy: {score}")
        scores.append(score)  # append accuracy
    else:
        # tranf. with 3 models
        axtime = 0
        # score = train_multimodal_dense(trn, val, no_features, epochs_count, val_lbls, trn_lbls, 0.001, tst, tst_lbls)
        score = train_multimodal_transformer(Xtrain, Xval, no_features, epochs_count, labels_val, labels_train, 0.001, Xtest, labels_test, axtime, dims_of_reshape)
        print(f"Accuracy: {score}")
        scores.append(score)  # append accuracy
    # # trying xgboost as agg. model
    # clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=1000)
    # clf.fit(trn, labels_train)
    # score = clf.score(tst, labels_test)
    # print(f"Accuracy: {score}")
    # scores.append(score)  # append accuracy

    # getting values from activations and then to another model

    # transformer
    # # Xtrain = np.transpose(Xtrain, (0, 2, 1))
    # # Xtest = np.transpose(Xtest, (0, 2, 1))
    # input_shape = Xtrain.shape[1:]
    # model = build_model(
    #     input_shape,
    #     head_size=256,
    #     num_heads=4,
    #     ff_dim=4,
    #     num_transformer_blocks=4,
    #     mlp_units=[150, 100, 40, 10],  # these are after the multihead attention so for classification so would add these dense layers
    #     n_classes=n_classes,
    #     mlp_dropout=0.4,
    #     dropout=0.25,
    # )
    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #     metrics=["accuracy"],
    # )
    # model.summary()
    # # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)] # if we want to stop after certain epochs if there is no improvement
    # model.fit(
    #     stats.zscore(Xtrain, axis=0),
    #     labels,
    #     # validation_split=0.2, # because we have validation externally
    #     epochs=50,
    #     batch_size=64,
    #     # callbacks=callbacks,
    # )

    # predictions = model.predict(stats.zscore(Xtest, axis=0))
    # score = accuracy_score(np.argmax(labels_test, axis=1), np.argmax(predictions, axis=1))
    # print(f"Accuracy: {score}")
    # scores.append(score)  # append accuracy


s = 1
print(scores)
print(np.mean(scores))
print(np.std(scores))


# %%
if preprocess == 0:
    # dat = np.load('x_train_face.npy')
    co = 0
    pathm = './seed/SEED_IV/eeg_feature_smooth/1/'
    for file in tqdm(os.listdir(pathm)):
        co += 1
        if co == 23:
            break
        path = './deap/data_preprocessed_matlab/' + file
        files.append(file)  # could try to get the names from here and see certain condition ..

        data = scipy.io.loadmat(path)
        fs = 128
        timev = np.arange(0, 60, 1/fs)

        trials = data['data']
        # removing baseline (3sec.)
        trials = trials[:, :, fs*3:]
        # trials to 2d so that we could read and epoch it
        trials2d = []
        for i in range(trials.shape[0]):
            trials2d.append(trials[i, :, :])

        trials2d = np.concatenate(trials2d, axis=1)

        # min-max scaling ... so the min will be zero and the max will be one ...
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(trials2d)
        trials2d = scaler.transform(trials2d)

        labels = data['labels']  # cont. value for each of them

        ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
                    'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
                    'hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Resp', 'Plethysmog', 'Temp']  # 'T3', 'T4', 'T5', 'T6'

        info = mne.create_info(ch_names=ch_names,
                               sfreq=fs,
                               ch_types='eeg')

        raw = mne.io.RawArray(trials2d, info)

        # segmenting into 2sec. segments
        temp = []
        temp_labels = []
        for i in range(0, trials.shape[-1], window_seconds*fs):
            temp_labels.append(labels)
            temp.append(trials[:, :32, i:i+window_seconds*fs])
        trials = np.concatenate(temp, axis=0)
        labels = np.concatenate(temp_labels, axis=0)

        # TF with sim. power at 30Hz
        # signal = np.cos(2. * np.pi * 10. * raw.times) * 1000
        # trials2d[0] = signal
        # raw = mne.io.RawArray(trials2d, info)

        # # Create an event matrix: 3 columns, first is index of event, second is unused, third is event id.
        # events = np.array([[(i)*trials.shape[2], 0, 1] for i in range(trials.shape[0])])
        # epochs = mne.Epochs(raw, events, tmax=60-(1/fs), tmin=0, baseline=None, preload=True)
        # ss = epochs.get_data() # this should be like trials .. ok

        # # Compute ERP
        # channelsOfAnalysis = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
        #                       'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

        # channel_no = [0, 2, 16, 19]  # only taking these four channels .. according to the code from github
        # channelsOfAnalysis = ['Fp1', 'F3', 'Fp2', 'F4']

        # epochs.pick_channels(channelsOfAnalysis)
        # Reorder the channels ... so that it would be the same order as channelsOfAnalysis
        # epochs.reorder_channels(channelsOfAnalysis)

        # t = [nearest(timev, -3), nearest(timev, 0)]
        # np.mean(erp.data[:, t[0]:t[1]], axis=1)

        # all_trials.append(power)
        # all_labels.append(labels)
        # trials = epochs.get_data()
        all_trials.append(trials)
        all_labels.append(labels)


# plt.plot(np.squeeze(dat[0, 1, :])) # one trial
# plt.plot(np.mean(dat[:,:,0:384], axis = 0))
# plt.plot(np.mean(dat[:, :, 384:384*2], axis=0))

# plt.plot(np.squeeze(np.mean(dat, axis=(0, 1)))) # erp .. the average is small but signals look fine perhaps ..

# erp = epochs.average()
# fig, ax = plt.subplots()
# sig = np.mean(erp.data, axis=0)
# ax.plot(timev, sig)
# ax.set(title='ERP', ylabel='Amplitude', xlabel='Time (s)')
# plt.show()

# all_trials = np.load('all_trials.npy', allow_pickle=True)  # ppnts trials ch freq time
# all_labels = np.load('all_labels.npy', allow_pickle=True)

# temp = []
# temp_labels = []
# for i in range(all_trials.shape[0]):  # all_trials.shape[0]
#     temp.append(all_trials[i, :, :, :, :])
#     temp_labels.append(all_labels[i, :, :])
# temp = np.concatenate(temp, axis=0)
# all_trials = temp
# labels = np.concatenate(temp_labels, axis=0)
# del temp, temp_labels

# s = 1

# all_trials = all_trials[:, (0, 2, 16, 19), :, :]
# aggregating data from ppnts with video recordings as well
trials = np.concatenate(all_trials, axis=0)
labels = np.concatenate(all_labels, axis=0)

# %% LSTM
# kf = StratifiedKFold(n_splits=5)  # 5 fold and stratified so that there should be deviants with each split ..
# kf.get_n_splits(trials)  # Returns the number of splitting iterations in the cross-validator

# // labels = [int(_) for _ in (labels[:, 0] > labels[:, 1])]  # valence: 1 .. arousal: 0 ... no because it is not valence vs arousal but it is classification for each

labels = [int(_) for _ in (labels[:, 0] > 5)]  # high valence: 1 .. low valence: 0
labels = np.array(labels)


# n_folds = 5
# kfold = KFold(n_splits=n_folds, shuffle=True)
# scores = []
# all_trials = np.transpose(all_trials, (0, 2, 3, 1))
# for train, val in tqdm(kfold.split(all_trials, labels)):
#     # cnn
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(210, 210, 4)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     model.fit(all_trials[train], labels[train], epochs=10, verbose=1)
#     y_pred = model.predict(all_trials[val])
#     acc = accuracy(labels[val], y_pred)
#     print(f"Accuracy: {acc}")
#     scores.append(acc)

# # with transfer learning ... no because these were trained on coloured image for a different objective ..
# input_tensor = Input(shape=(210, 210, 32))
# # 1x1 conv. to reduce channels
# x = Conv2D(3, (1, 1))(input_tensor)
# base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=x)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
# for layer in base_model.layers:
#     layer.trainable = False
# model.compile(optimizer='rmsprop', loss='binary_crossentropy')
# model.fit(all_trials[train], labels[train], epochs=10, batch_size=32)
# # Unfreeze all layers of the Inception V3 model
# for layer in base_model.layers:
#     layer.trainable = True
# model.compile(optimizer='SGD', loss='binary_crossentropy')
# model.fit(all_trials[train], labels[train], epochs=10, batch_size=32)
# y_pred = model.predict(all_trials[val])
# acc = accuracy(labels[val], y_pred)
# print(f"Accuracy: {acc}")
# scores.append(acc)

# for train_index, test_index in kf.split(trials, labels):
#     X_train, X_test = trials[train_index], trials[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]
#     # trying with lstm
#     X_train, X_test = np.transpose(X_train, (0, 2, 1)), np.transpose(X_test, (0, 2, 1))  # 3d trials time channels
#     n_features = X_train.shape[2]

#     model = Sequential()
#     model.add(LSTM(16, input_shape=(None, n_features), return_sequences=True))
#     model.add(LeakyReLU())
#     model.add(LSTM(8, return_sequences=True))
#     # model.add(LeakyReLU())
#     # model.add(GlobalMaxPooling1D())
#     # model.add(LSTM(6))
#     model.add(LeakyReLU())
#     model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=[accuracy])
# model.fit(X_train, y_train, epochs=50, verbose=1)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# acc = accuracy(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"Mean Absolute Error: {mae}")
# print(f"Accuracy: {acc}")


# %% getting power and then decision trees
data = np.transpose(trials, (0, 2, 1))
power = np.empty((data.shape[0], data.shape[1]//2+1, data.shape[2]))  # because until half the time points
freqs = np.fft.fftfreq(data.shape[1], 1/fs)
freqs = freqs[:power.shape[1]]

for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        # fft = fftpack.fft(data[i, :, j])
        # power[i, :, j] = np.abs(fft[:fft.shape[0]//2+1])**2
        freqs, power[i, :, j] = welch(data[i, :, j], fs=fs, nperseg=data.shape[1])

# power = power[:, nearest(freqs, 4):nearest(freqs, 45), :]
# freqs = freqs[nearest(freqs, 4):nearest(freqs, 45)]

freq_hold = freqs
delta = [.5, 4]
theta = [4, 8]
alpha = [8, 12]
beta = [13, 35]
gamma = [30, 40]

delta_pw = np.mean(power[:, nearest(freq_hold, delta[0]):nearest(freq_hold, delta[1])+1, :], axis=(1))
theta_pw = np.mean(power[:, nearest(freq_hold, theta[0]):nearest(freq_hold, theta[1])+1, :], axis=(1))
alpha_pw = np.mean(power[:, nearest(freq_hold, alpha[0]):nearest(freq_hold, alpha[1])+1, :], axis=(1))
beta_pw = np.mean(power[:, nearest(freq_hold, beta[0]):nearest(freq_hold, beta[1])+1, :], axis=(1))
gamma_pw = np.mean(power[:, nearest(freq_hold, gamma[0]):nearest(freq_hold, gamma[1])+1, :], axis=(1))

X_all = np.column_stack((delta_pw, theta_pw, alpha_pw, beta_pw, gamma_pw))


# %% leave one out
lo_scores = []
for i in range(0, X_all.shape[0], int(40*(60/window_seconds))):
    print(i)
    Xtest = X_all[i:i+int(40*(60/window_seconds))]
    labels_test = labels[i:i+int(40*(60/window_seconds))]

    X = np.concatenate((X_all[:i], X_all[i+int(40*(60/window_seconds)):]), axis=0)
    labels_train = np.concatenate((labels[:i], labels[i+int(40*(60/window_seconds)):]), axis=0)

    s = 1

    # plotting to check difference in power
    # fig = plt.figure(figsize=(20, 10))
    # plt.plot(freqs, np.mean(power[np.squeeze(labels == 0), :, :], axis=(0, 2)))
    # plt.plot(freqs, np.mean(power[np.squeeze(labels == 1), :, :], axis=(0, 2)))
    # plt.legend(('arousal', 'valence'))
    # plt.xlabel("frequency (Hz")
    # plt.ylabel("power")
    # plt.title("power spectrum")
    # plt.show()

    # random forest
    # n_folds = 5
    # kfold = KFold(n_splits=n_folds, shuffle=True)
    scores = []
    # power = np.squeeze(np.mean(power, axis=2))
    # X = power.reshape(power.shape[0], -1)

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    # PCA
    # pca = PCA(n_components=100)
    # X = pca.fit_transform(X)
    # for train, val in tqdm(kfold.split(X, labels)):
    # clf = RandomForestClassifier(n_estimators=1000)
    # clf = LinearSVC()
    # scaler = StandardScaler()
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=500)
    clf.fit(X, labels_train)
    score = clf.score(Xtest, labels_test)
    print(score)
    scores.append(score)  # append accuracy

    # model = Sequential()
    # model.add(Dense(150))
    # model.add(LeakyReLU())
    # model.add(Dense(100))
    # model.add(LeakyReLU())
    # model.add(Dense(40))
    # model.add(LeakyReLU())
    # model.add(Dense(10))
    # model.add(LeakyReLU())
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=[accuracy])
    # scaler = StandardScaler()
    # model.fit(X, labels_train, epochs=150, verbose=1)
    # y_pred = model.predict(Xtest)
    # # mse = mean_squared_error(labels[val], y_pred)
    # # mae = mean_absolute_error(labels[val], y_pred)
    # acc = accuracy(labels_test, y_pred)
    # # print(f"Mean Squared Error: {mse}")
    # # print(f"Mean Absolute Error: {mae}")
    # print(f"Accuracy: {acc}")
    # scores.append(acc)

    # print(scores)

print(scores)
s = 1

# %% the model on the training set
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=2000)
clf.fit(X, labels)
# with open('model.pkl', 'wb') as file:
# pickle.dump(clf, file)
# np.save('freq.npy', freqs)
clf.save_model('model.json')
