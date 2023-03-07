import pandas as pd
import os
import sys
import warnings
import time
import random
import numpy as np

import argparse

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

sys.path.append("../")
from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
sys.path.insert(1, "../exp")


def load_data():
    random.seed(42)
    partition_num_list = random.sample(range(1000), 10)
    base_dir = f'../exp_data/Churn_Prediction/partitions/p{partition_num_list[0]}'
    data = pd.read_csv(f'{base_dir}/train.csv')
    members = pd.read_csv(f'{base_dir}/members.csv')
    transactions = pd.read_csv(f'{base_dir}/transactions.csv')
    logs = pd.read_csv(f'{base_dir}/logs.csv')

    for i in range(1, len(partition_num_list)):
        base_dir = f'../exp_data/Churn_Prediction/partitions/p{partition_num_list[i]}'
        data = pd.concat([data, pd.read_csv(f'{base_dir}/train.csv')])
        members = pd.concat([members, pd.read_csv(f'{base_dir}/members.csv')])
        transactions = pd.concat([transactions, pd.read_csv(f'{base_dir}/transactions.csv')])
        logs = pd.concat([logs, pd.read_csv(f'{base_dir}/logs.csv')])
   
    members = members.drop(columns=['gender'])
    transaction_member = transactions.merge(members, how='inner', left_on='msno', right_on='msno')
    all_logs = logs.merge(transaction_member, how='inner', left_on='msno', right_on='msno')
    all_logs = all_logs.dropna()

    all_logs['transaction_date'] = all_logs['transaction_date'].astype(int)
    all_logs['membership_expire_date'] = all_logs['membership_expire_date'].astype(int)
    all_logs['registration_init_time'] = all_logs['registration_init_time'].astype(int)

    X = data.copy()
    label = X.pop('is_churn')

    train_logs = X.merge(all_logs, how='inner', left_on='msno', right_on='msno')
    train_logs = train_logs.dropna()

    X_train, X_rem, y_train, y_rem = train_test_split(X, label, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_logs



def evaluate_test_data(
    train_data, train_labels, test_data, test_labels, optimal_query_list, ml_model="rf"
):
    for query in optimal_query_list:
        # arg_list = []
        # for key in query["param"]:
        #     arg_list.append(query["param"][key])
        new_feature, join_keys = sqlgen_task.generate_new_feature(arg_dict=query["param"])
        train_data = train_data.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
        test_data = test_data.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    if 'msno' in train_data.columns.tolist():
        train_data = train_data.drop(columns=['msno'])
    if 'msno' in test_data.columns.tolist():
        test_data = test_data.drop(columns=['msno'])
    if ml_model == 'xgb':
        clf = XGBClassifier(random_state = 0)
    elif ml_model == 'rf':
        clf = RandomForestClassifier(random_state=0, class_weight='balanced')
    elif ml_model == 'lr':
        clf = LinearRegression(random_state = 0)
    elif ml_model == 'nn':
        clf = MLPClassifier(random_state=0)

    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    score = roc_auc_score(test_labels, predictions)

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--ml_model', type=str, default='rf', help = "ML model",required = False)
    args = parser.parse_args()
    ml_model = args.ml_model

    train_data, train_labels, valid_data, valid_labels, test_data, test_labels, user_log = load_data()
    seed_list = [0, 42, 89, 550, 572, 1024, 3709]
    test_score_list = []

    print(len(user_log['is_auto_renew'].unique()))
    print(len(user_log['transaction_date'].unique()))
    print(len(user_log['membership_expire_date'].unique()))
    print(len(user_log['is_cancel'].unique()))

    fkeys = ["msno"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "STDDEV_POP"]

    # agg_attrs = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100',
    #              'num_unq', 'total_secs', 'payment_plan_days', 'actual_amount_paid', ]
    # predicate_attrs = ['payment_method_id', 'is_auto_renew', 'transaction_date',
    #                    'membership_expire_date', 'is_cancel', 'city', 'bd', 'registered_via', 'registration_init_time']

    agg_attrs = ['num_25', 'num_50', 'num_75', 'num_985', 'num_100',
                 'num_unq', 'total_secs', 'payment_plan_days', 'actual_amount_paid', 
                 'payment_method_id', 'is_auto_renew', 'transaction_date',
                 'membership_expire_date', 'is_cancel', 'city', 'bd', 
                 'registered_via', 'registration_init_time']
    # predicate_attrs = ['payment_method_id', 'is_auto_renew', 'transaction_date',
    #                    'membership_expire_date', 'is_cancel', 'city', 'bd', 'registered_via', 'registration_init_time']
    random.seed(42)
    predicate_attrs = random.sample(agg_attrs, 15)
    print(predicate_attrs)
    
    groupby_keys = fkeys
    predicate_attr_types = {
        'transaction_date': {
            "type": "datetime",
            "choices": [str(x) for x in user_log['transaction_date'].unique()] + ["None"],
        },
        'membership_expire_date': {
            "type": "datetime",
            "choices": [str(x) for x in user_log['membership_expire_date'].unique()] + ["None"],
        },
        'registration_init_time': {
            "type": "datetime",
            "choices": [str(x) for x in user_log['registration_init_time'].unique()] + ["None"],
        },
    }
    for attr in predicate_attrs:
        if attr in ['transaction_date', 'membership_expire_date', 'registration_init_time']:
            continue
        if user_log[attr].dtype == np.dtype('int64'):
            predicate_attr_types[attr] = {
                "type": "categorical",
                "choices": [str(x) for x in user_log[attr].unique()],
            }
        else:
            predicate_attr_types[attr] = {
                "type": "float",
                "low": min(user_log[attr].unique()),
                "high": max(user_log[attr].unique())
            }
    # predicate_attr_types = {
    #     'payment_method_id': {
    #         "type": "categorical",
    #         "choices": [str(x) for x in user_log['payment_method_id'].unique()] + ["None"],
    #     },
    #     'is_auto_renew': {
    #         "type": "categorical",
    #         "choices": [str(x) for x in user_log['is_auto_renew'].unique()] + ["None"],
    #     },
    #     'transaction_date': {
    #         "type": "datetime",
    #         "choices": [str(x) for x in user_log['transaction_date'].unique()] + ["None"],
    #     },
    #     'membership_expire_date': {
    #         "type": "datetime",
    #         "choices": [str(x) for x in user_log['membership_expire_date'].unique()] + ["None"],
    #     },
    #     'is_cancel': {
    #         "type": "categorical",
    #         "choices": [str(x) for x in user_log['is_cancel'].unique()] + ["None"],
    #     },
    #     'city': {
    #         "type": "categorical",
    #         "choices": [str(x) for x in user_log['city'].unique()] + ["None"],
    #     },
    #     'bd': {
    #         "type": "categorical",
    #         "choices": [str(x) for x in user_log['bd'].unique()] + ["None"],
    #     },
    #     'registered_via': {
    #         "type": "categorical",
    #         "choices": [str(x) for x in user_log['registered_via'].unique()] + ["None"],
    #     },
    #     'registration_init_time': {
    #         "type": "datetime",
    #         "choices": [str(x) for x in user_log['registration_init_time'].unique()] + ["None"],
    #     },
    # }
    

    time_list = []
    all_optimal_query_list = []
    for seed in seed_list[:]:
        start = time.time()
        query_template = QueryTemplate(
            fkeys=fkeys,
            agg_funcs=agg_funcs,
            agg_attrs=agg_attrs,
            all_predicate_attrs=predicate_attrs,
            predicate_attrs=[],
            groupby_keys=groupby_keys,
            all_predicate_attrs_type=predicate_attr_types,
            predicate_attrs_type={},
        )

        sqlgen_task = SQLGen()
        sqlgen_task.build_task(
            query_template=query_template,
            base_table=train_data,
            labels=train_labels,
            valid_table=valid_data,
            valid_labels=valid_labels,
            test_table=test_data,
            test_labels=test_labels,
            relevant_table=user_log,
        )

        print(f'seed: {seed}')
        optimal_query_list = sqlgen_task.optimize(
            ml_model=ml_model,
            #metric="f1",
            outer_budget=5,
            mi_budget=5000,
            mi_topk=100,
            base_tpe_budget=400,
            turn_on_mi=True,
            turn_on_mapping_func=False,
            seed=seed
        )
        print((seed, optimal_query_list))
        all_optimal_query_list.append((seed, optimal_query_list))
        test_score = evaluate_test_data(
            train_data, train_labels, test_data, test_labels, optimal_query_list, ml_model=ml_model
        )
        print(f"Test score of seed {seed}: {test_score}")
        test_score_list.append(test_score)
        end = time.time()
        print(f"Running Time: {end - start}")
        time_list.append(end - start)
    for query_list in all_optimal_query_list:
        print(query_list)
    for single_time in time_list:
        print(single_time)