import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import log_loss
from airbnb_project.preprocessing import preprocessing_data


def train_model():
    main_df = preprocessing_data()

    # Model params
    model_lgb = lgb.LGBMClassifier(
        objective='muticlass',
        metric='multi_logloss',
        num_class=11,
        n_jobs=-1,
        n_estimators=100000,
        learning_rate=0.001,
        num_leaves=200,
        max_depth=-1,
        feature_fraction=0.9,
        bagging_freq=5,
        bagging_fraction=0.9,
        min_data_in_leaf=100,
        silent=-1,
        verbose=-1,
        max_bin=300,
        bagging_seed=11,
    )

    # Split train and test data
    y = main_df['country_destination']
    features = [f for f in main_df.columns if f not in ['id', 'user_id', 'country_destination']]
    X = main_df[features]

    # X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # Reindex the data
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # initialize KFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    logloss_scores = []
    models = []  # save model for each fold
    for i, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print('...... training {}th fold \n'.format(i + 1))
        tr_X = X_train.loc[train_idx]
        tr_y = y_train.loc[train_idx]

        va_X = X_train.loc[valid_idx]
        va_y = y_train.loc[valid_idx]

        model = model_lgb
        model.fit(tr_X, tr_y, eval_set=[(tr_X, tr_y), (va_X, va_y)], eval_metric='multi_logloss', verbose=500,
                  early_stopping_rounds=300)

        # calculate current logloss after training the model
        pred_va_y = model.predict_proba(va_X, num_iteration=model.best_iteration_)
        logloss = log_loss(va_y, pred_va_y)
        print('current best logloss is:{}'.format(logloss))
        logloss_scores.append(logloss)
        models.append(model)

    print('the average mean logloss is:{}'.format(np.mean(logloss_scores)))
