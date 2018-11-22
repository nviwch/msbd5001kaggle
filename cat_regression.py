import numpy as np
import pandas as pd
import csv
from catboost import CatBoostRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate

def data_preprocessing(df):
    # drop useless information
    df.drop('id', axis=1, inplace=True)
    df.drop('random_state', axis=1, inplace=True)
    # df.drop('scale', axis=1, inplace=True)
    df.drop('n_informative', axis=1, inplace=True)
    df.drop('flip_y', axis=1, inplace=True)
    df.drop('n_clusters_per_class', axis=1, inplace=True)
    df.drop('alpha', axis=1, inplace=True)

def data_augmentation(df):
    augment_df = df.copy()

    # integer gaussian noise
    # noise = np.round(np.random.normal(0,2,df.shape[0])).astype(int)
    # augment_df['max_iter'] = df['max_iter'] + noise
    # noise = np.round(np.random.normal(0,2,df.shape[0])).astype(int)
    # augment_df['n_samples'] = df['n_samples'] + noise
    # noise = np.round(np.random.normal(0,1,df.shape[0])).astype(int)
    # augment_df['n_features'] = df['n_features'] + noise

    # real gaussian noise
    # percentage_noise = (np.round(np.random.normal(0,2,df.shape[0])) / 100) + 1
    # augment_df['flip_y'] = df['flip_y'] * percentage_noise
    percentage_noise = (np.round(np.random.normal(0,2,df.shape[0])) / 100) + 1
    augment_df['time'] = df['time'] * percentage_noise

    return augment_df

# load csv
original_train_df = pd.read_csv("train.csv")
test_x = pd.read_csv("test.csv")

data_preprocessing(original_train_df)
data_preprocessing(test_x)

original_train_df['time'] = original_train_df['time'] * 10e5 / (original_train_df['n_samples'] * original_train_df['max_iter'])
valid_size = 100

test_recover_array = np.full(test_x.shape[0], 10e5)
test_n_samples_array = test_x['n_samples'].values
test_max_iter_array = test_x['max_iter'].values
test_recover_array = np.divide(np.divide(test_recover_array, test_n_samples_array), test_max_iter_array)

test_x.drop(['n_samples', 'max_iter'], axis=1, inplace=True)

prediction_list = np.arange(test_x.shape[0])
MSE_list = []

for j in range(1):
    # original_train_df = original_train_df.sample(frac=1).reset_index(drop=True)

    recover_array = np.full(valid_size, 10e5)
    n_samples_array = original_train_df[:valid_size]['n_samples'].values
    max_iter_array = original_train_df[:valid_size]['max_iter'].values
    recover_array = np.divide(np.divide(recover_array, n_samples_array), max_iter_array)

    dropped_train_df = original_train_df.drop(['n_samples', 'max_iter'], axis=1)

    dropped_train_df['n_jobs'].replace(-1, 8, inplace=True)  
    test_x['n_jobs'].replace(-1, 8, inplace=True)  
    test_x['n_jobs'].replace(16, 8, inplace=True)  

    valid_df = dropped_train_df[:valid_size]
    train_df = dropped_train_df[valid_size:]

    complete_augment_df = train_df.copy()
    for i in range(10):
        augment_df = data_augmentation(train_df)
        complete_augment_df = complete_augment_df.append(augment_df, ignore_index=True)

    train_x = complete_augment_df.drop('time', axis=1)
    train_y = complete_augment_df['time']

    valid_x = valid_df.drop('time', axis=1)
    valid_y = valid_df['time']

    categorical_features_indices = [0]
    cat = CatBoostRegressor(logging_level='Verbose', cat_features=categorical_features_indices, 
                            use_best_model=True, 
                            random_state=1,
                            od_type='IncToDec')

    cat.fit(train_x, train_y, eval_set=(valid_x, valid_y),cat_features=categorical_features_indices, use_best_model=True)

    feature_importance = cat.get_feature_importance()
    print(dict(zip(list(train_x), feature_importance)))

    valid_y_array = np.divide(valid_y.values, recover_array)

    valid_y_predicted = cat.predict(valid_x)
    valid_y_predicted = np.divide(valid_y_predicted, recover_array)
    Cat_MSE = metrics.mean_squared_error(valid_y_array, valid_y_predicted)
    print('Cat MSE:', Cat_MSE)

    MSE_list.append(Cat_MSE)

    cat_test_y_predicted = cat.predict(test_x)
    cat_test_y_predicted = np.divide(cat_test_y_predicted, test_recover_array)

    # with open('predictedtime' + str(j) + '.csv', "w") as output:
    #     writer = csv.writer(output, lineterminator='\n')
    #     for val in cat_test_y_predicted:
    #         writer.writerow([val])

    prediction_list = np.column_stack((prediction_list,cat_test_y_predicted))

np.savetxt("prediction_list_" + str(8) + "_" + str(np.average(Cat_MSE)) + "_.csv", prediction_list, delimiter=",")
np.savetxt("MSE_list_" + str(8) + "_" + str(np.average(Cat_MSE)) +  "_.csv", np.array(MSE_list), delimiter=",")

# with open('predictedtime.csv', "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in cat_test_y_predicted:
#         writer.writerow([val])