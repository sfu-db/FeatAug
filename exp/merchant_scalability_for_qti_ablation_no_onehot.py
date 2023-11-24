import pandas as pd
import os
import sys
import warnings
import time
import random
import numpy as np
import math
import copy

import argparse
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append("../")

# from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
# from sqlgen.sqlgen_without_query_template_identification import QueryTemplate, SQLGen
# from sqlgen.sqlgen_random import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_spearman_proxy import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_lr_proxy import QueryTemplate, SQLGen
# from sqlgen.sqlgen_scalability_analysis import QueryTemplate, SQLGen
from feataug.feataug_scalability_analysis_no_onehot import QueryTemplate, SQLGen

sys.path.insert(1, "../exp")

COL_DUPLICATE_NUM = 0
SAMPLE_COL_NUM = 4

def load_data():
    train = pd.read_csv("../exp_data/Merchant_Category_Rec/train.csv")
    merchants = pd.read_csv("../exp_data/Merchant_Category_Rec/merchants.csv")
    historical_transactions = pd.read_csv("../exp_data/Merchant_Category_Rec/historical_transactions.csv")
    new_transactions = pd.read_csv("../exp_data/Merchant_Category_Rec/new_merchant_transactions.csv")

    random.seed(42)
    sampled_card_id = random.sample(train['card_id'].unique().tolist(), 50000)
    sampled_train = train[train['card_id'].isin(sampled_card_id)]
    sampled_histocial_transactions = historical_transactions[historical_transactions['card_id'].isin(sampled_card_id)]
    sampled_new_transactions = new_transactions[new_transactions['card_id'].isin(sampled_card_id)]
    sampled_merchant_id = sampled_histocial_transactions['merchant_id'].unique().tolist()

    sampled_merchants = merchants[merchants['merchant_id'].isin(sampled_merchant_id)]
    sampled_merchants = sampled_merchants.drop_duplicates(subset=['merchant_id'])
    sampled_histocial_transactions = sampled_histocial_transactions.merge(sampled_merchants, how='left',
                                                                          on='merchant_id')
    all_columns = sampled_histocial_transactions.columns
    log_to_drop = [x for x in all_columns if x.endswith('_dup')]
    sampled_histocial_transactions = sampled_histocial_transactions.drop(columns=log_to_drop)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in all_columns:
        if column in ['card_id', 'purchase_date']:
            continue
        if column in log_to_drop:
            continue
        elif sampled_histocial_transactions[column].dtype not in ['float64', 'int64']:
            print(column)
            encoded_labels = le.fit_transform(sampled_histocial_transactions[column])
            sampled_histocial_transactions[column] = encoded_labels
    sampled_histocial_transactions['purchase_date'] = pd.to_datetime(
        sampled_histocial_transactions['purchase_date']).astype(int) // 10 ** 9
    sampled_train['first_active_month'] = pd.to_datetime(sampled_train['first_active_month']).astype(int) // 10 ** 9

    train_labels = sampled_train.pop('target')
    sampled_train = sampled_train.fillna(0)
    sampled_histocial_transactions = sampled_histocial_transactions.fillna(0)

    X_train, X_rem, y_train, y_rem = train_test_split(sampled_train, train_labels, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    # Original DataFrame
    duplicated_sampled_historical_transactions = sampled_histocial_transactions.copy()
    df_final = duplicated_sampled_historical_transactions.copy()

    # Copy and rename columns for the first duplication
    for i in range(COL_DUPLICATE_NUM):
        df1 = duplicated_sampled_historical_transactions.copy()
        df1.columns = [f'{col}{i}' for col in duplicated_sampled_historical_transactions.columns]

        # Concatenate all DataFrames horizontally
        df_final = pd.concat([df_final, df1], axis=1)
        print(i)
    print(df_final.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, df_final


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
        clf = XGBRegressor(random_state = 0)
    elif ml_model == 'rf':
        clf = RandomForestRegressor(random_state=0)
    elif ml_model == 'lr':
        clf = LinearRegression(random_state = 0)
    elif ml_model == 'nn':
        clf = MLPRegressor(random_state=0)

    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    score = mean_squared_error(test_labels, predictions)
    score = -1 * (math.sqrt(score))

    return score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--ml_model', type=str, default='rf', help = "ML model",required = False)
    args = parser.parse_args()
    ml_model = args.ml_model

    train_data, train_labels, valid_data, valid_labels, test_data, test_labels, user_log = load_data()
    seed_list = [0, 42, 89, 550, 572, 1024, 3709, 97, 119]
    test_score_list = []

    fkeys = ["card_id"]
    # fkeys = ["user_id"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "ENTROPY", "KURTOSIS", "MODE", "MAD", "MEDIAN"]
    pre_agg_attrs = ['authorized_flag', 'city_id_x', 'category_1_x',
       'installments', 'category_3', 'merchant_category_id_x', 'merchant_id',
       'month_lag', 'purchase_amount', 'purchase_date', 'category_2_x',
       'state_id_x', 'subsector_id_x', 'merchant_group_id',
       'merchant_category_id_y', 'subsector_id_y', 'numerical_1',
       'numerical_2', 'category_1_y', 'most_recent_sales_range',
       'most_recent_purchases_range', 'avg_sales_lag3', 'avg_purchases_lag3',
       'active_months_lag3', 'avg_sales_lag6', 'avg_purchases_lag6',
       'active_months_lag6', 'avg_sales_lag12', 'avg_purchases_lag12',
       'active_months_lag12', 'category_4', 'city_id_y', 'state_id_y',
       'category_2_y']
    agg_attrs = copy.deepcopy(pre_agg_attrs)
    for i in range(COL_DUPLICATE_NUM):
        agg_attrs.append(f'{col}{i}')
        # for col in pre_agg_attrs:
        #     if not (col.startswith('merchant_group_id') or col.startswith('merchant_category_id_x') or col.startswith('merchant_category_id_y')):
        #         agg_attrs.append(f'{col}{i}')
            # agg_attrs.append(f'{col}{i}')
        # agg_attrs += [f'{col}{i}' for col in pre_agg_attrs]
    random.seed(42)
    predicate_attrs = random.sample(agg_attrs, 15)
    # predicate_attrs = []
    # while len(predicate_attrs) < SAMPLE_COL_NUM:
    #     col = random.choice(agg_attrs)
    # for col in agg_attrs:
    #     if not col in predicate_attrs:
    #         predicate_attrs.append(col)
    print(len(predicate_attrs))
    print(predicate_attrs)

    # predicate_attrs = ['time_stamp']
    # predicate_attrs = []
    groupby_keys = fkeys
    predicate_attr_types = {
        'purchase_date': {
            "type": "datetime",
            "choices": [str(x) for x in user_log['purchase_date'].unique()] + ["None"],
        }
    }

    for attr in predicate_attrs:
        if not attr in ['authorized_flag', 'category_1_x', 'category_3', 'merchant_id', 'category_1_y', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
            if user_log[attr].dtype == 'int64':
                predicate_attr_types[attr] = {
                    "type": "categorical",
                    "choices": [str(x) for x in user_log[attr].unique()] + ["None"],
                }
            elif user_log[attr].dtype == 'float64':
                predicate_attr_types[attr] = {
                    "type": "float",
                    "low": min(user_log[attr].unique()),
                    "high": max(user_log[attr].unique())
                }
        else:
            predicate_attr_types[attr] = {
                "type": "categorical",
                "choices": [str(x) for x in user_log[attr].unique()] + ["None"],
            }

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

        # with all optimizations
        print(f'seed: {seed}')
        optimal_query_list = sqlgen_task.optimize(
            ml_model=ml_model,
            metric='root_mean_squared_error',
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
        #     metric='root_mean_squared_error',
        #     outer_budget=5,
        #     mi_budget=200,
        #     mi_topk=0,
        #     base_tpe_budget=90,
        #     turn_on_mi=False,
        #     turn_on_mapping_func=False,
        #     seed=seed,
        #     query_template_num=8
        # )

        print((seed, optimal_query_list))
        all_optimal_query_list.append((seed, optimal_query_list))
        print(f"Query Template Identification Time: {sqlgen_task.qti_time}")
        print(f"Warm up Time: {sqlgen_task.warmup_time}")
        print(f"Search Time: {sqlgen_task.search_time}")
        # test_score = evaluate_test_data(
        #     train_data, train_labels, test_data, test_labels, optimal_query_list, ml_model=ml_model
        # )
        # print(f"Test score of seed {seed}: {test_score}")
        # test_score_list.append(test_score)
        end = time.time()
        print(f"Running Time: {end - start}")
        time_list.append(end - start)
    for query_list in all_optimal_query_list:
        print(query_list)
    for single_time in time_list:
        print(single_time)