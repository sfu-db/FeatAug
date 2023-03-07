import pandas as pd
import os
import sys
import time
import warnings
import copy
import numpy as np
import featuretools as ft
import argparse

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

sys.path.append("../")
from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
sys.path.insert(1, "../exp")

def load_data():
    path = "../exp_data/Instacart/data"
    orders = pd.read_csv(os.path.join(path, "orders.csv"))
    order_products_train = pd.read_csv(os.path.join(path, "order_products__train.csv"))
    order_products_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"))
    products = pd.read_csv(os.path.join(path, "products.csv"))
    departments = pd.read_csv(os.path.join(path, "departments.csv"))

    departments['department'] = departments['department'].astype('category')
    orders['eval_set'] = orders['eval_set'].astype('category')
    products['product_name'] = products['product_name'].astype('category')

    log_prior = order_products_prior \
        .merge(orders[orders.eval_set == 'prior'], how='inner', on='order_id') \
        .merge(products, how='inner', on='product_id') \
        .merge(departments, how='inner', on='department_id')

    log_train = order_products_train \
        .merge(orders[orders.eval_set == 'train'], how='inner', on='order_id') \
        .merge(products, how='inner', on='product_id') \
        .merge(departments, how='inner', on='department_id')

    new_product_name = log_train['product_name'].str.contains(r'Banana')
    log_train = log_train.drop('product_name', 1)
    log_train['product_name'] = new_product_name
    log_train['product_name'] = log_train['product_name'].astype(int)

    train_data = log_train.groupby('user_id')['product_name'].sum().reset_index()
    train_data["product_name"] = np.where(train_data["product_name"] <= 0, 0, 1)

    train_data_sample = train_data.sample(n=10000, random_state=42)
    users_sample = train_data_sample['user_id'].tolist()
    log_prior_sample = log_prior[log_prior['user_id'].isin(users_sample)]

    new_product_name_prior = log_prior_sample['product_name'].str.contains(r'Banana')
    log_prior_sample = log_prior_sample.drop('product_name', 1)
    log_prior_sample['product_name'] = new_product_name_prior
    log_prior_sample['product_name'] = log_prior_sample['product_name'].astype(int)

    log_prior_sample = log_prior_sample.drop(columns=['eval_set', 'aisle_id', 'department'])
  
    X = copy.deepcopy(train_data_sample)
    label = X.pop('product_name')

    X_train, X_rem, y_train, y_rem = train_test_split(X, label, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, log_prior_sample


def evaluate_test_data(
    train_data, train_labels, test_data, test_labels, optimal_query_list, ml_model="rf"
):
    for query in optimal_query_list:
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

    fkeys = ["user_id"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "STDDEV_POP"]
    agg_attrs = ['order_id', 'product_id', 'reordered', 'user_id', 'order_number', 'department_id', 'product_name']
    predicate_attrs = [ 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'department_id', 'product_id', 'reordered', 'user_id']

    groupby_keys = fkeys
    predicate_attr_types = {
        "order_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["order_id"].unique()],
        },
        "product_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["product_id"].unique()],
        },
        "reordered": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["reordered"].unique()],
        },
        "user_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["user_id"].unique()],
        },
        "order_number": {
            "type": "int",
            "low": min(user_log["order_number"].unique()),
            "high": max(user_log["order_number"].unique())
        },
        "order_dow": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["order_dow"].unique()],
        },
        "order_hour_of_day": {
            "type": "int",
            "low": min(user_log["order_hour_of_day"].unique()),
            "high": max(user_log["order_hour_of_day"].unique())
        },
        "days_since_prior_order": {
            "type": "int",
            "low": min(user_log["days_since_prior_order"].unique()),
            "high": max(user_log["days_since_prior_order"].unique())
        },
        "department_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["department_id"].unique()] + ["None"],
        },
    }

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