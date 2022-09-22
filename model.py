import sys
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    ### Initialize console parametrs
    train_path, test_path, save_path = str(sys.argv[1])[:-1], str(sys.argv[2])[:-1], str(sys.argv[3])

    ### Split sets on features/ target
    train = pd.read_csv(train_path)
    train_X = train.drop(['target_full_ltv_day30', 'target_sub_ltv_day30', 'target_ad_ltv_day30', 'target_iap_ltv_day30'], axis=1)
    trian_y = train['target_full_ltv_day30'].copy()

    test = pd.read_csv(test_path)
    test_X = test.drop(['target_full_ltv_day30', 'target_sub_ltv_day30', 'target_ad_ltv_day30', 'target_iap_ltv_day30'], axis=1)
    test_y = test['target_full_ltv_day30'].copy()

    ### Create ColumnTransformer
    cat = [col for col in train_X.columns if train_X[col].dtype == "O"]
    num = [col for col in train_X.columns if train_X[col].dtype != "O"]

    cols_trans = ColumnTransformer(transformers=[
        ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=False), cat),
        ('scaling', StandardScaler(), num)
    ])

    ### Initialize models
    rfr = RandomForestRegressor()
    etr = ExtraTreesRegressor()
    gbr = GradientBoostingRegressor()
    estimators = [('rfr', rfr), ('etr', etr), ('gbr', gbr)]

    ### Create pipeline
    pipe = Pipeline(
        steps=[
            ('data_engineering', cols_trans), 
            ('imputing', KNNImputer(n_neighbors=7)), 
            ('clustering', KMeans(n_clusters=2)), 
            ('stacking_regressor', StackingRegressor(estimators=estimators, cv=3, n_jobs=-1, verbose=1))
        ],
        verbose=1
    )

    ### Train model
    pipe.fit(train_X, trian_y)

    ### Do generalization 
    prediction = pipe.predict(test_X)

    ### Save results
    save_file = pd.DataFrame(prediction, columns=['target_full_ltv_day30'], index=test.index)
    save_file.to_csv(save_path)

    ### Show metric
    print()
    print('-' * 50)
    print(f'MSE: {mean_squared_error(test_y, prediction)}')