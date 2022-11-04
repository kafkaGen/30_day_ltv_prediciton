import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgb


if __name__ == '__main__':
    # Initialize console parametrs
    train_path, test_path, save_path = str(
        sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    # Split sets on features/ target
    train = pd.read_csv(train_path)
    train_X = train.drop(['target_full_ltv_day30', 'target_sub_ltv_day30',
                         'target_ad_ltv_day30', 'target_iap_ltv_day30'], axis=1)
    train_y = train['target_full_ltv_day30'].copy()

    test = pd.read_csv(test_path)
    test_X = test.drop(['target_full_ltv_day30', 'target_sub_ltv_day30',
                       'target_ad_ltv_day30', 'target_iap_ltv_day30'], axis=1)
    test_y = test['target_full_ltv_day30'].copy()

    print()
    print('-' * 50)
    print('Data loaded')

    # make list of our categorical features
    cat = [col for col in train_X.columns if train_X[col].dtype == "O"]
    # make list of our numerical features
    num = [col for col in train_X.columns if train_X[col].dtype != "O"]
    # make one hot encoding of categorical features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    train_cat = ohe.fit_transform(train_X.loc[:, cat])
    test_cat = ohe.transform(test_X.loc[:, cat])
    train_X = np.concatenate(
        (train_X.loc[:, num].to_numpy(), train_cat), axis=1)
    test_X = np.concatenate((test_X.loc[:, num].to_numpy(), test_cat), axis=1)

    print()
    print('-' * 50)
    print('Data prepared')

    # find our clusters
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
    train_X_clst = kmeans.fit_predict(train_X)
    test_X_clst = kmeans.predict(test_X)

    print()
    print('-' * 50)
    print('Clusters detected')

    # prepare our prediction DataFrame
    pred = test_y.copy()
    # build model
    model = lgb.LGBMRegressor(
        objective='mae', n_jobs=-2, random_state=42, verbose=2)
    for clst in np.arange(kmeans.n_clusters):
        # find elements of same clusters
        train_idx = np.argwhere(train_X_clst == clst).flatten()
        test_idx = np.argwhere(test_X_clst == clst).flatten()
        # fit model
        model.fit(train_X[train_idx], np.log(
            train_y + 1)[train_idx])
        # make prediciton
        pred[test_idx] = model.predict(test_X[test_idx])

    print()
    print('-' * 50)
    print('Model trained')

    # Save results
    save_file = pd.DataFrame(pred, columns=['target_full_ltv_day30'])
    save_file.to_csv(save_path, index=False)

    # Show metric
    print()
    print('-' * 50)
    print(f'MAE score: {mean_absolute_error(np.log(test_y + 1), pred)}')
    print(f'R2 score: {r2_score(np.log(test_y + 1), pred)}')
