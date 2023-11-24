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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

sys.path.append("../")
#from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
#from sqlgen.sqlgen_with_mapping_func import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_spearman_proxy import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_pearson_proxy import QueryTemplate, SQLGen
# from sqlgen.sqlgen_random import QueryTemplate, SQLGen
# from sqlgen.sqlgen_without_query_template_identification import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_lr_proxy import QueryTemplate, SQLGen 
from feataug.feataug_scalability_analysis_no_onehot import QueryTemplate, SQLGen

sys.path.insert(1, "../exp")

COL_DUPLICATE_NUM = 0
SAMPLE_COL_NUM = 4


# def load_data():
#     # Load all dataset
#     path = "../exp_data/Instacart"
#     order_products = pd.read_csv(os.path.join(path, "order_products__prior.csv"))
#     orders = pd.read_csv(os.path.join(path, "orders.csv"))
#     departments = pd.read_csv(os.path.join(path, "departments.csv"))
#     products = pd.read_csv(os.path.join(path, "products.csv"))
#
#     user = pd.DataFrame()
#     user["user_id"] = orders["user_id"]
#     user = user.drop_duplicates(keep="first", inplace=False)
#
#     log = (
#         orders.merge(order_products)
#         .merge(products)
#         .sort_values(by=["user_id", "order_number"])
#     )
#     print("number of users:", len(user), ", number of logs:", len(log))
#
#     orders_train = orders[orders["eval_set"] == "train"]
#     global user_train
#     user_train = pd.DataFrame()
#     user_train["user_id"] = orders_train["user_id"]
#     log_train = (
#         user_train.merge(orders)
#         .merge(order_products)
#         .merge(products)
#         .sort_values(by=["user_id", "order_number"])
#     )
#     print("training users:", len(user_train), "training data:", len(log_train))
#
#     orders_test = orders[orders["eval_set"] == "test"]
#     user_test = pd.DataFrame()
#     user_test["user_id"] = orders_test["user_id"]
#     log_test = (
#         user_test.merge(orders)
#         .merge(order_products)
#         .merge(products)
#         .sort_values(by=["user_id", "order_number"])
#     )
#     print("test users:", len(user_test), "test data:", len(log_test))
#
#     new_product_name = log_train["product_name"].str.contains(r"Banana")
#     log_train = log_train.drop("product_name", 1)
#     log_train["product_name"] = new_product_name
#     log_train = log_train[
#         [
#             "user_id",
#             "order_id",
#             "product_id",
#             "eval_set",
#             "order_number",
#             "order_dow",
#             "order_hour_of_day",
#             "days_since_prior_order",
#             "add_to_cart_order",
#             "reordered",
#             "aisle_id",
#             "department_id",
#             "product_name",
#         ]
#     ]
#     log_train["product_name"] = log_train["product_name"].astype(int)
#     log_train = log_train.fillna(0)
#     print(
#         "# Banana:",
#         len(log_train[log_train["product_name"] == True]),
#         "# non-Banana:",
#         len(log_train[log_train["product_name"] == False]),
#     )
#
#     # Make Labels
#     training = log_train[log_train["eval_set"] == "train"]
#     has_Banana = {}
#     for i, row in training.iterrows():
#         if row[-1] == True:
#             has_Banana.update({row[0]: 1})
#     print("# of users buy banana", len(has_Banana))
#
#     user_train["label"] = np.nan
#     for item in has_Banana.items():
#         user_train.loc[user_train["user_id"] == item[0], "label"] = 1
#
#     # fillna with 0
#     user_train = user_train.fillna(0)
#     print("neg samples:", len(user_train[user_train["label"] == 0]))
#     print("pos samples:", len(user_train[user_train["label"] == 1]))
#     user_train.head()
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         user_train.drop(["label"], axis=1),
#         user_train["label"],
#         test_size=0.2,
#         random_state=42,
#     )
#
#     print(X_train.shape, X_test.shape)
#     train_data = X_train
#     train_labels = y_train
#     test_data = X_test
#     test_labels = y_test
#
#     log_train["index"] = (
#         log_train["user_id"].astype(str)
#         + "_"
#         + log_train["order_id"].astype(str)
#         + "_"
#         + log_train["product_id"].astype(str)
#     )
#     log_train = log_train[
#         [
#             "index",
#             "user_id",
#             "order_id",
#             "product_id",
#             "product_name",
#             "department_id",
#             "order_dow",
#             "days_since_prior_order",
#         ]
#     ]
#
#     log_train_copy = copy.deepcopy(log_train)
#     log_train_copy = log_train_copy.rename({"index": "index_copy"}, axis=1)
#     log_train_copy["unique_id"] = range(0, len(log_train_copy))
#
#     from woodwork.logical_types import Categorical, Integer, Datetime
#
#     log_train_vtypes = {
#         "index": Categorical,
#         "user_id": Categorical,
#         "order_id": Categorical,
#         "product_id": Categorical,
#         "order_dow": Categorical,
#         "days_since_prior_order": Categorical,
#         "department_id": Categorical,
#         "product_name": Categorical,
#     }
#
#     es = ft.EntitySet("instacart")
#     es.add_dataframe(
#         dataframe_name="log_train",
#         dataframe=log_train,
#         index="index",
#         logical_types=log_train_vtypes,
#     )
#
#     es.normalize_dataframe(
#         base_dataframe_name="log_train", new_dataframe_name="users", index="user_id"
#     )
#
#     feature_matrix, features = ft.dfs(
#         target_dataframe_name="users",
#         agg_primitives=["sum", "min", "max", "count", "mean"],
#         #                                   where_primitives = ["count"],
#         trans_primitives=[],
#         ignore_columns={"log_train": ["order_id", "product_id", "department_id"]},
#         entityset=es,
#         verbose=True,
#     )
#
#     train_data = train_data.merge(feature_matrix, on="user_id", how="left")
#     test_data = test_data.merge(feature_matrix, on="user_id", how="left")
#
#     train_data = train_data.reindex(sorted(train_data.columns), axis=1)
#     test_data = test_data.reindex(sorted(test_data.columns), axis=1)
#
#     return train_data, train_labels, test_data, test_labels, log_train

# def load_data():
#     path = "../exp_data/Instacart/data"
#     orders = pd.read_csv(os.path.join(path, "orders.csv"))
#     order_products_train = pd.read_csv(os.path.join(path, "order_products__train.csv"))
#     order_products_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"))
#     products = pd.read_csv(os.path.join(path, "products.csv"))
#     departments = pd.read_csv(os.path.join(path, "departments.csv"))
#
#     departments['department'] = departments['department'].astype('category')
#     orders['eval_set'] = orders['eval_set'].astype('category')
#     products['product_name'] = products['product_name'].astype('category')
#
#     log_prior = order_products_prior \
#         .merge(orders, how='inner', on='order_id') \
#         .merge(products, how='inner', on='product_id') \
#         .merge(departments, how='inner', on='department_id')
#     log_train = order_products_train \
#         .merge(orders, how='inner', on='order_id') \
#         .merge(products, how='inner', on='product_id') \
#         .merge(departments, how='inner', on='department_id')
#     sampled_user_id = log_train['user_id'].sample(n=2000).tolist()
#
#     import random
#     random.seed(42)
#     train_users = random.sample(sampled_user_id, int(len(sampled_user_id) * 0.8))
#     test_users = [user_id not in train_users for user_id in train_users]
#
#     train_data = log_train[log_train['user_id'].isin(train_users)].loc[:, ('user_id', 'product_id', 'reordered')]
#     train_labels = train_data.loc[:, ('reordered')]
#     train_data = train_data.loc[:, ('user_id', 'product_id')]
#
#     test_data = log_train[log_train['user_id'].isin(test_users)].loc[:, ('user_id', 'product_id', 'reordered')]
#     test_labels = test_data.loc[:, ('reordered')]
#     test_data = test_data.loc[:, ('user_id', 'product_id')]
#
#     log_prior_for_train = log_prior[log_prior['user_id'].isin(train_users)]
#     log_prior_for_train = log_prior_for_train.drop(columns=['eval_set', 'product_name', 'department', 'reordered'])
#
#     return train_data, train_labels, test_data, test_labels, log_prior_for_train

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
    # log_train = log_train[['user_id', 'order_id', 'product_id', 'eval_set', 'order_number',
    #                       'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order',
    #                        'reordered', 'aisle_id', 'department_id', 'product_name']]
    log_train['product_name'] = log_train['product_name'].astype(int)

    train_data = log_train.groupby('user_id')['product_name'].sum().reset_index()
    train_data["product_name"] = np.where(train_data["product_name"] <= 0, 0, 1)

    train_data_sample = train_data.sample(n=50000, random_state=42)
    users_sample = train_data_sample['user_id'].tolist()
    log_prior_sample = log_prior[log_prior['user_id'].isin(users_sample)]

    new_product_name_prior = log_prior_sample['product_name'].str.contains(r'Banana')
    log_prior_sample = log_prior_sample.drop('product_name', 1)
    # log_prior_sample['product_name'] = new_product_name_prior
    # log_prior_sample['product_name'] = log_prior_sample['product_name'].astype(int)

    log_prior_sample = log_prior_sample.drop(columns=['eval_set', 'aisle_id', 'department'])
    # X_train, X_test, y_train, y_test = train_test_split(
    #     train_data_sample.drop(['product_name'], axis=1), train_data_sample['product_name'], test_size=0.2,
    #     random_state=42)

    X = copy.deepcopy(train_data_sample)
    label = X.pop('product_name')

    X_train, X_rem, y_train, y_rem = train_test_split(X, label, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, log_prior_sample

def load_data_with_featuretools():
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels, all_data, all_labels, user_log = load_data()
    user_log_to_return = copy.deepcopy(user_log)
    user_log = user_log.reset_index()
    es = ft.EntitySet(id='instacart')
    es = es.add_dataframe(dataframe_name="all_data",
                      index='user_id',
                      dataframe=all_data)
    es = es.add_dataframe(dataframe_name="user_log",
                      index='index',
                      dataframe=user_log)
    es = es.add_relationship(
        parent_dataframe_name="all_data",
        parent_column_name='user_id',
        child_dataframe_name="user_log",
        child_column_name='user_id'
    )
    fm, features = ft.dfs(entityset=es,
                      target_dataframe_name='all_data', 
                      agg_primitives=['max','time_since_last','any','mode','count','first',
                                      'skew','n_most_common','num_unique','time_since_first',
                                      'all','min','last','mean','percent_true','std','entropy',
                                      'median','avg_time_between','sum','num_true','trend'],
                      verbose=False)
    fm = fm.fillna(0)
    fm_train = train_data.merge(fm, how='left', on='user_id')
    fm_valid = valid_data.merge(fm, how='left', on='user_id')
    fm_test = test_data.merge(fm, how='left', on='user_id')
    fm_train = fm_train.fillna(0)
    fm_valid = fm_valid.fillna(0)
    fm_test = fm_test.fillna(0)

    return fm_train, train_labels, fm_valid, valid_labels, fm_test, test_labels, user_log_to_return

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
    seed_list = [0, 42, 89, 550, 572, 1024, 3709, 119, 97]
    test_score_list = []

    fkeys = ["user_id"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "ENTROPY", "KURTOSIS", "MODE", "MAD", "MEDIAN"]
    agg_attrs = ['order_id', 'product_id', 'reordered', 'user_id', 'order_number', 'department_id']
    predicate_attrs = [ 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'department_id', 'product_id', 'reordered', 'user_id']

    # fkeys = ["user_id", "product_id"]
    # agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "STDDEV_POP"]
    # agg_attrs = ["user_id", 'order_id', 'product_id', 'order_dow', 'department_id' , 'days_since_prior_order']
    # predicate_attrs = ['days_since_prior_order', 'order_dow', 'department_id']

    print("Department ID length:")
    print(len([str(x) for x in user_log["department_id"].unique()] + ["None"]))
    print("Order Number:")
    print(len([str(x) for x in user_log["order_number"].unique()] + ["None"]))
    print("days_since_prior_order")
    print(len([str(x) for x in user_log["days_since_prior_order"].unique()] + ["None"]))

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
        # "product_name": {
        #     "type": "categorical",
        #     "choices": [str(x) for x in user_log["product_name"].unique()] + ["None"],
        # },
        # "aisle_id": {
        #     "type": "categorical",
        #     "choices": [str(x) for x in user_log["aisle_id"].unique()] + ["None"],
        # },
        "department_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["department_id"].unique()] + ["None"],
        },
    }

    # query_template_size = 1
    # for attr in predicate_attrs:
    #     if predicate_attr_types[attr]["type"] == 'categorical':
    #         query_template_size = query_template_size * len(predicate_attr_types[attr]["choices"])
    #     elif predicate_attr_types[attr]["type"] == 'datetime':
    #         query_template_size = query_template_size * len(predicate_attr_types[attr]["choices"]) * len(predicate_attr_types[attr]["choices"])
    #     elif predicate_attr_types[attr]["type"] == 'int':
    #         query_template_size = query_template_size * int(predicate_attr_types[attr]["high"] - predicate_attr_types[attr]["low"])
    #     elif predicate_attr_types[attr]["type"] == 'float':
    #         query_template_size = query_template_size * int(int(predicate_attr_types[attr]["high"] - predicate_attr_types[attr]["low"]) / 0.1)
    # print(f"Max number of queries each query template: {query_template_size}")

    time_list = []
    all_optimal_query_list = []
    for seed in seed_list[1:2]:
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

        # With all optimizations
        print(f'seed: {seed}')
        optimal_query_list = sqlgen_task.optimize(
            ml_model=ml_model,
            #metric="f1",
            outer_budget=5,
            mi_budget=200,
            mi_topk=50,
            base_tpe_budget=40,
            turn_on_mi=True,
            turn_on_mapping_func=False,
            seed=seed,
            query_template_num=8
        )

        # Ablation study: without mi
        # print(f'seed: {seed}')
        # optimal_query_list = sqlgen_task.optimize(
        #     ml_model=ml_model,
        #     #metric="f1",
        #     outer_budget=5,
        #     mi_budget=5000,
        #     mi_topk=0,
        #     base_tpe_budget=80,
        #     turn_on_mi=False,
        #     turn_on_mapping_func=False,
        #     seed=seed,
        #     query_template_num=8
        # )
        
        print((seed, optimal_query_list))
        print(f"Query Template Identification Time: {sqlgen_task.qti_time}")
        print(f"Warm up Time: {sqlgen_task.warmup_time}")
        print(f"Search Time: {sqlgen_task.search_time}")
        # all_optimal_query_list.append((seed, optimal_query_list))
        # test_score = evaluate_test_data(
        #     train_data, train_labels, test_data, test_labels, optimal_query_list, ml_model=ml_model
        # )
        # print(f"Test score of seed {seed}: {test_score}")
        #test_score_list.append(test_score)
        end = time.time()
        print(f"Running Time: {end - start}")
        time_list.append(end - start)
    for query_list in all_optimal_query_list:
        print(query_list)
    for single_time in time_list:
        print(single_time)