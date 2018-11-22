from keras import callbacks, regularizers
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import seaborn as sb
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold
import matplotlib.pyplot as plt

def data_preprocessing(df):
    # drop useless information
    df.drop('id', axis=1, inplace=True)
    df.drop('random_state', axis=1, inplace=True)
    df.drop('scale', axis=1, inplace=True)
    # df.drop('n_informative', axis=1, inplace=True)
    df.drop('flip_y', axis=1, inplace=True)
    # df.drop('l1_ratio', axis=1, inplace=True)
    # df.drop('n_clusters_per_class', axis=1, inplace=True)
    # df.drop('alpha', axis=1, inplace=True)
    # data impute
    df['n_jobs'].replace(-1, 8, inplace=True) 
    df['n_jobs'].replace(16, 8, inplace=True)  

def strong_data_augmentation(df):
    augment_df = df.copy()

    scale = np.random.uniform(1,8,df.shape[0]).astype(int)
    noise = np.round(np.random.normal(0,3,df.shape[0])).astype(int)
    percentage_noise = (np.round(np.random.normal(0,3,df.shape[0])) / 100) + 1
    augment_df['n_samples'] = df['n_samples'] * scale + noise
    augment_df['time'] = df['time'] * scale * percentage_noise

    scale = np.random.uniform(1,8,df.shape[0]).astype(int)
    percentage_noise = (np.round(np.random.normal(0,3,df.shape[0])) / 100) + 1
    augment_df['max_iter'] = df['max_iter'] * scale
    augment_df['time'] = df['time'] * scale * percentage_noise

    # scale = np.random.uniform(1,8,df.shape[0]).astype(int)
    # percentage_noise = (np.round(np.random.normal(0,3,df.shape[0])) / 100) + 1
    # augment_df['n_jobs'] = df['n_jobs'] * scale
    # augment_df['time'] = df['time'] / scale * percentage_noise

    # integer gaussian noise
    # for column in ['max_iter']:
    #     noise = np.round(np.random.normal(0,3,df.shape[0])).astype(int)
    #     augment_df[column] = df[column] + noise

    # real gaussian noise
    for column in ['flip_y', 'scale', 'l1_ratio']:
        percentage_noise = (np.round(np.random.normal(0,3,df.shape[0])) / 100) + 1
        augment_df[column] = df[column] * percentage_noise

    return augment_df


def weak_data_augmentation(df):
    augment_df = df.copy()

    # integer gaussian noise
    # for column in ['n_features']:
    #     noise = np.round(np.random.normal(0,3,df.shape[0])).astype(int)
    #     augment_df[column] = df[column] + noise

    # real gaussian noise
    for column in ['time', 'n_informative', 'l1_ratio', 'flip_y']:
        percentage_noise = (np.round(np.random.normal(0,3,df.shape[0])) / 100) + 1
        augment_df[column] = df[column] * percentage_noise

    return augment_df

def encode_penalty(df):
    df['l2_ratio'] = 1 - df['l1_ratio']
    df.loc[df['penalty'] == 'l1', 'l2_ratio'] = 0
    df.loc[df['penalty'] == 'l2', 'l1_ratio'] = 0
    df.loc[df['penalty'] == 'none', ['l1_ratio', 'l2_ratio']] = 0
    df.drop('penalty', axis=1, inplace=True)

def gen_recovery_array(n_row, arr, scale=10e5):
    recover_array = np.full(n_row, scale)
    for arr_i in arr:
        recover_array = np.divide(recover_array, arr_i)
    recover_array = np.reshape(recover_array, (-1,1)) 
    return recover_array

def neural_regressor(n_dimen):
    model = Sequential()
    model.add(Dense(32, kernel_initializer='normal', activation='relu', input_dim=n_dimen))
    model.add(Dropout(0.4))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

# load csv
original_train_df = pd.read_csv("train.csv")
test_x = pd.read_csv("test.csv")

data_preprocessing(original_train_df)
data_preprocessing(test_x)

# C_mat = original_train_df.corr()
# fig = plt.figure(figsize = (12,12))
# sb.heatmap(C_mat, vmax=0.8, vmin=-0.8, cmap="PiYG", square=True)
# plt.show()

# negative_df = original_train_df[original_train_df['n_jobs'] == -1]
# original_train_df = original_train_df[original_train_df['n_jobs'] != -1]

# original_train_df = original_train_df.sample(frac=1, random_state=1).reset_index(drop=True)

k = 10e4
original_train_df['time'] = original_train_df['time'] * k / (original_train_df['n_samples'] * original_train_df['max_iter'])
valid_size = 100

recover_array = gen_recovery_array(valid_size, [original_train_df[:valid_size]['n_samples'].values, original_train_df[:valid_size]['max_iter'].values], k)
test_recover_array = gen_recovery_array(test_x.shape[0], [test_x['n_samples'].values, test_x['max_iter'].values], k)
original_train_df.drop(['n_samples', 'max_iter'], axis=1, inplace=True)
test_x.drop(['n_samples', 'max_iter'], axis=1, inplace=True)

scale_df = original_train_df.copy()
encode_penalty(scale_df)
encode_penalty(test_x)
scale_train_x = scale_df.drop('time', axis=1)
scale_train_y = scale_df['time']
Xscaler = MinMaxScaler()
X_concat = pd.concat([scale_train_x, test_x])
Xscaler.fit(X_concat)
Yscaler = MinMaxScaler()
Y = np.reshape(scale_train_y.values, (-1,1))
Yscaler.fit(Y)

valid_df = original_train_df[:valid_size]
train_df = original_train_df[valid_size:]

# Data Augmentation
complete_augment_df = weak_data_augmentation(train_df)
for i in range(499):
    augment_df = weak_data_augmentation(train_df)
    complete_augment_df = complete_augment_df.append(augment_df, ignore_index=True)

encode_penalty(complete_augment_df)
encode_penalty(valid_df)

# C_mat = complete_augment_df.corr()
# fig = plt.figure(figsize = (12,12))
# sb.heatmap(C_mat, vmax=0.8, vmin=-0.8, cmap="PiYG", square=True)
# plt.show()

# encode_penalty(negative_df)

train_x = complete_augment_df.drop('time', axis=1)
train_y = complete_augment_df['time']

valid_x = valid_df.drop('time', axis=1)
valid_y = valid_df['time']

# ng_x = negative_df.drop('time', axis=1)
# ng_y = negative_df['time']

# Xscaler = StandardScaler()
# X_concat = pd.concat([train_x, valid_x])
# Xscaler.fit(X_concat)
# Yscaler = StandardScaler()
# Y_concat = pd.concat([train_y, valid_y])
# Y = np.reshape(Y_concat.values, (-1,1))
# Yscaler.fit(Y)

train_x = Xscaler.transform(train_x)
train_y = Yscaler.transform(np.reshape(train_y.values, (-1,1)))
valid_x = Xscaler.transform(valid_x)
valid_y = Yscaler.transform(np.reshape(valid_y.values, (-1,1)))
test_x = Xscaler.transform(test_x)

# model = load_model('model-013-0.04-0.13.h5')

# for n_thread in range(1, 128):
#     ng_x['n_jobs'] = n_thread

#     # Standard scaler
#     ng_x_transformed = Xscaler.transform(ng_x)
#     ng_y_transformed = Yscaler.transform(np.reshape(ng_y.values, (-1,1)))
#     # test_x = scaler.transform(test_x)

#     ng_y_transformed = Yscaler.inverse_transform(ng_y_transformed)
#     ng_y_predicted = model.predict(ng_x_transformed)
#     ng_y_predicted = Yscaler.inverse_transform(ng_y_predicted)
#     print(str(n_thread) + ':NN train MSE:', metrics.mean_squared_error(ng_y_transformed, ng_y_predicted))

#     # train_y = Yscaler.inverse_transform(train_y)
#     # train_y_predicted = model.predict(train_x)
#     # train_y_predicted = Yscaler.inverse_transform(train_y_predicted)
#     # print('NN train MSE:', metrics.mean_squared_error(train_y, train_y_predicted))

#     # valid_y_predicted = model.predict(valid_x)
#     # print('NN true MSE:', metrics.mean_squared_error(valid_y, valid_y_predicted))
#     # valid_y = Yscaler.inverse_transform(valid_y)
#     # valid_y_predicted = Yscaler.inverse_transform(valid_y_predicted)
#     # print('NN true MSE:', metrics.mean_squared_error(valid_y, valid_y_predicted))

model = neural_regressor(train_x.shape[1])
print(model.summary())

# # model.load_weights('model-098-0.56-0.33.h5', by_name=True)

checkpoint = callbacks.ModelCheckpoint('model-{epoch:03d}-{loss:.2f}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
history = model.fit(train_x, train_y, batch_size = 128, epochs = 50, validation_data=(valid_x, valid_y), callbacks=[checkpoint])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# model = load_model('model-002-0.00-0.00.h5')

# valid_y_predicted = model.predict(valid_x)
# print('NN scaled MSE:', metrics.mean_squared_error(valid_y, valid_y_predicted))
# valid_y = Yscaler.inverse_transform(valid_y)
# valid_y_predicted = Yscaler.inverse_transform(valid_y_predicted)
# print('NN divided MSE:', metrics.mean_squared_error(valid_y, valid_y_predicted))
# valid_y = np.divide(valid_y, recover_array)
# valid_y_predicted = np.divide(valid_y_predicted, recover_array)
# print('NN true MSE:', metrics.mean_squared_error(valid_y, valid_y_predicted))

# nn_test_y_predicted = model.predict(test_x)
# nn_test_y_predicted = Yscaler.inverse_transform(nn_test_y_predicted)
# nn_test_y_predicted = np.divide(nn_test_y_predicted, test_recover_array)
# with open('nn_predictedtime.csv', "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in nn_test_y_predicted:
#         writer.writerow(val)

plt.show()