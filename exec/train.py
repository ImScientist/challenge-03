import os
import json
import datetime
import argparse
import lightgbm as lgb
from ta.preprocess import get_data
from ta.training import get_feature_names, get_loss_fct_weights
from typing import Dict

DATA_DIR = os.environ['DATA_DIR']
RESULTS_DIR = os.environ['RESULTS_DIR']


def train(
        data_dir: str,
        results_dir: str,
        t_train: datetime,
        t_val: datetime,
        lgb_parameters: Dict
):
    os.makedirs(results_dir, exist_ok=True)

    data, le, le_countries = get_data(
        data_dir=data_dir,
        load_data=True,
        save_data=False,
        countries_overwrite=False,
        nrows=None,
        google_api_key=None
    )

    contiuous_columns, categorical_columns, target_col, _ = \
        get_feature_names(data)

    train_cond = data['ts'] < t_train
    val_cond = data['ts'].between(t_train, t_val)
    # test_cond = data['ts'] > t_val

    weights_train = get_loss_fct_weights(data, train_cond, target_col)
    weights_val = get_loss_fct_weights(data, val_cond, target_col)
    # weights_test = get_loss_fct_weights(data, test_cond, target_col)

    lgb_train = lgb.Dataset(
        data=data.loc[train_cond, contiuous_columns + categorical_columns],
        label=data.loc[train_cond, target_col],
        weight=weights_train,
        categorical_feature=categorical_columns,
        free_raw_data=False)

    lgb_val = lgb.Dataset(
        data=data.loc[val_cond, contiuous_columns + categorical_columns],
        label=data.loc[val_cond, target_col],
        weight=weights_val,
        categorical_feature=categorical_columns,
        free_raw_data=False)

    # lgb_test = lgb.Dataset(
    #     data=data.loc[test_cond, contiuous_columns + categorical_columns],
    #     label=data.loc[test_cond, target_col],
    #     weight=weights_test,
    #     # feature_name=features,
    #     categorical_feature=categorical_columns,
    #     free_raw_data=False)

    evals_result = dict()

    gbm = lgb.train(
        params=lgb_parameters,
        train_set=lgb_train,
        early_stopping_rounds=50,
        valid_names=['train', 'val'],
        valid_sets=[lgb_train, lgb_val],
        verbose_eval=20,
        evals_result=evals_result)

    lgb_best_score = gbm.best_score

    summary = dict(
        lgb_n_trees=gbm.num_trees(),
        lgb_params=lgb_parameters,
        lgb_best_score=lgb_best_score,
        lgb_feature_importance=dict(
            features=gbm.feature_name(),
            importance_split=gbm.feature_importance(importance_type='split').tolist(),
            importance_gain=gbm.feature_importance(importance_type='gain').tolist(),
        ),
        features=dict(
            features_categorical=contiuous_columns,
            features_continuous=categorical_columns,
            target=target_col
        ),
        other_params=dict(),
        evals_result=evals_result
    )

    # save the model
    gbm.save_model(os.path.join(results_dir, 'model.txt'))

    # save the summary
    if summary is not None:
        with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent='\t')

    print('DONE')


if __name__ == "__main__":
    """ Train a model by using `$DATA_DIR/data.pkl` as training dataset.
    
    Example: 
        source .env
        python exec/train.py
    """
    t_train = datetime.datetime(2017, 4, 29)
    t_val = datetime.datetime(2017, 5, 1)
    # t_test = datetime.datetime(2017, 5, 4)

    lgb_parameters = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary', 'auc'],
        'first_metric_only': True,

        'num_iterations': 500,
        'num_leaves': 11,
        'min_data_in_leaf': 30,

        'learning_rate': 0.03,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 1,

        'lambda_l2': 0.1
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, dest="data_dir", default=DATA_DIR,
                        help="Data directory. The preprocessed data will be stored here, as well.")

    parser.add_argument("--results_dir", type=str, dest="results_dir", default=RESULTS_DIR,
                        help="Directory where all training artifacts will be saved.")

    args = parser.parse_args()

    for arg in vars(args):
        print("{0:34s} \t {1:20s}".format(arg, str(getattr(args, arg))))

    train(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        t_train=t_train,
        t_val=t_val,
        lgb_parameters=lgb_parameters
    )
