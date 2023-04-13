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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append("../")

from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen

sys.path.insert(1, "../exp")


# def load_data():
#     path = "../exp_data/Tmall/"
#     train_data = pd.read_csv(os.path.join(path, "train_data.csv"))
#     test_data = pd.read_csv(os.path.join(path, "test_data.csv"))
#     user_log = pd.read_csv(os.path.join(path, "user_log.csv"))
#     # print(len(user_log))
#
#     train_data = train_data.drop(train_data.columns[0], axis=1)
#     train_labels = train_data["label"]
#     train_data = train_data.drop(columns=["label"])
#
#     test_data = test_data.drop(test_data.columns[0], axis=1)
#     test_labels = test_data["label"]
#     test_data = test_data.drop(columns=["label"])
#
#     user_log = user_log.drop(user_log.columns[0], axis=1)
#     # data = pd.concat([train_data, test_data])
#     # print(len(user_log))
#     # print("Train true label:", len(train_data[train_data['label'] == 1]), 'Train false label:',
#     #       len(train_data[train_data['label'] == 0]))
#     # print("Test true label:", len(test_data[test_data['label'] == 1]), 'Test false label:',
#     #       len(test_data[test_data['label'] == 0]))
#     print(test_labels.shape)
#     return train_data, train_labels, test_data, test_labels, user_log


def load_data():
    path = "../exp_data/Tmall/"
    train_data = pd.read_csv(os.path.join(path, "train_format1.csv"))
    user_log = pd.read_csv(os.path.join(path, "user_log_format1.csv"))
    user_info = pd.read_csv(os.path.join(path, "user_info_format1.csv"))
    user_log = user_log.rename(columns={"seller_id": "merchant_id"})
    
    print(len(train_data['user_id'].unique().tolist()))

    random.seed(42)
    sampled_users = random.sample(train_data['user_id'].unique().tolist(), 50000)
    print(sampled_users)
    train_data_sample = train_data[train_data['user_id'].isin(sampled_users)]
    user_log_sample = user_log[user_log['user_id'].isin(sampled_users)]
    user_info_sample = user_info[user_info['user_id'].isin(sampled_users)]
    user_log_sample = user_log_sample.merge(user_info_sample, how='inner', left_on='user_id', right_on='user_id')
    train_data_sample = train_data_sample.merge(user_info_sample, how='inner', left_on='user_id', right_on='user_id')
    for column in user_log_sample.columns:
        user_log_sample[column] = user_log_sample[column].fillna(0)

    train_labels_sample = train_data_sample["label"]
    train_data_sample = train_data_sample.drop(columns=["label"])
    train_data_sample = train_data_sample.fillna(0)

    X_train, X_rem, y_train, y_rem = train_test_split(train_data_sample, train_labels_sample, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, user_log_sample


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
    if ml_model == 'xgb':
        clf = XGBClassifier(random_state = 0)
    elif ml_model == 'rf':
        clf = RandomForestClassifier(random_state=0, class_weight='balanced')
    elif ml_model == 'lr':
        clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
    elif ml_model == 'nn':
        clf = MLPClassifier(random_state=0, alpha=0.01, learning_rate_init=0.01, max_iter=1000)

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

    fkeys = ["user_id", "merchant_id"]
    # fkeys = ["user_id"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "ENTROPY", "KURTOSIS", "MODE", "MAD", "MEDIAN"]
    agg_attrs = ["merchant_id", "brand_id", "action_type", "age_range", "cat_id", "user_id"]
    predicate_attrs = ["time_stamp", "age_range", "action_type", "cat_id", "brand_id"]

    # predicate_attrs = ['time_stamp']
    # predicate_attrs = []
    groupby_keys = fkeys
    predicate_attr_types = {
        "gender": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["gender"].unique()],
        },
        "age_range": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["age_range"].unique()],
        },
        "action_type": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["action_type"].unique()],
        },
        "time_stamp": {
            "type": "datetime",
            "choices": [str(x) for x in user_log["time_stamp"].unique()] + ["None"]
            #'low': int(min(user_log['time_stamp'].unique())),
            #'high': int(max(user_log['time_stamp'].unique())),
        },
        "cat_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["cat_id"].unique()],
        },
        "brand_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["brand_id"].unique()],
        },
    }
    # query_template_size = 1
    # for attr in predicate_attrs:
    #     if predicate_attr_types[attr]["type"] == 'categorical':
    #         query_template_size = query_template_size * len(predicate_attr_types[attr]["choices"])
    #     elif predicate_attr_types[attr]["type"] == 'datetime':
    #         query_template_size = query_template_size * len(predicate_attr_types[attr]["choices"]) * len(predicate_attr_types[attr]["choices"])
    #     elif predicate_attr_types[attr]["type"] == 'int':
    #         query_template_size = query_template_size * int(predicate_attr_types[attr]["max"] - predicate_attr_types[attr]["min"])
    #     elif predicate_attr_types[attr]["type"] == 'float':
    #         query_template_size = query_template_size * int(int(predicate_attr_types[attr]["max"] - predicate_attr_types[attr]["min"]) / 0.1)
    # print(f"Max number of queries each query template: {query_template_size}")

    time_list = []
    all_optimal_query_list = []
    for seed in seed_list[:1]:
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
            seed=seed,
            query_template_num=8
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