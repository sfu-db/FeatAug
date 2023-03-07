import pandas as pd
import os
import sys
import warnings
import time
import random
import numpy as np
import argparse

sys.path.append("../")
sys.path.insert(1, "../exp")

from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


def load_data():
    random.seed(42)
    partition_num_list = random.sample(range(500), 10)
    base_dir = f'../exp_data/Default_Prediction/partitions/p{partition_num_list[0]}'
    users = pd.read_csv(f'{base_dir}/users.csv')
    logs = pd.read_csv(f'{base_dir}/users_log.csv')

    for i in range(1, len(partition_num_list)):
        base_dir = f'../exp_data/Default_Prediction/partitions/p{partition_num_list[i]}'
        users = pd.concat([users, pd.read_csv(f'{base_dir}/users.csv')])
        logs = pd.concat([logs, pd.read_csv(f'{base_dir}/users_log.csv')])

    all_columns = logs.columns
    for column in all_columns:
        if logs[column].isnull().sum() > 0.2 * len(logs):
            logs = logs.drop(columns=column)

    from sklearn.preprocessing import LabelEncoder
    for column in logs.columns:
        if column == 'customer_ID':
            continue
        if logs[column].dtype not in ('float64', 'int64'):
            le = LabelEncoder()
            encoded_labels = le.fit_transform(logs[column])
            logs[column] = encoded_labels
    logs = logs.fillna(0)

    label = users.pop('target')
    print(len(label))

    X_train, X_rem, y_train, y_rem = train_test_split(users, label, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, logs


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
    if 'customer_ID' in train_data.columns.tolist():
        train_data = train_data.drop(columns=['customer_ID'])
    if 'customer_ID' in test_data.columns.tolist():
        test_data = test_data.drop(columns=['customer_ID'])
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

    fkeys = ["customer_ID"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG" , "APPROX_COUNT_DISTINCT", "VAR_POP", "STDDEV_POP"]

    agg_attrs = ['S_2','P_2','D_39','B_1','B_2','R_1','S_3','D_41','B_3','D_44','B_4','D_45','B_5',
                 'R_2','D_47','D_48','B_6','B_7','B_8','D_51','B_9','R_3','D_52','P_3','B_10','S_5','B_11','S_6',
                 'D_54','R_4','S_7','B_12','S_8','D_55','B_13','R_5','D_58','B_14','D_59','D_60','D_61','B_15','S_11',
                'D_62','D_63','D_64','D_65','B_16','B_18','B_19','B_20','D_68','S_12','R_6','S_13','B_21','D_69',
                 'B_22','D_70','D_71','D_72','S_15','B_23','P_4','D_74','D_75','B_24','R_7','B_25','B_26','D_78',
                 'D_79','R_8','S_16','D_80','R_10','R_11','B_27','D_81','S_17','R_12','B_28','R_13','D_83','R_14',
                 'R_15','D_84','R_16','B_30','S_18','D_86','R_17','R_18','B_31','S_19','R_19','B_32','S_20','R_20',
                 'R_21','B_33','D_89','R_22','R_23','D_91','D_92','D_93','D_94','R_24','R_25','D_96','S_22','S_23',
                 'S_24','S_25','S_26','D_102','D_103','D_104','D_107','B_36','B_37','R_27','B_38','D_109','D_112',
                 'B_40','D_113','D_114','D_115','D_116','D_117','D_118','D_119','D_120','D_121','D_122','D_123',
                 'D_124','D_125','D_126','D_127','D_128','D_129','B_41','D_130','D_131','D_133','R_28','D_139','D_140',
                 'D_141','D_143','D_144','D_145']
    random.seed(42)
    predicate_attrs = random.sample(agg_attrs, 20)
    print(predicate_attrs)
    
    groupby_keys = fkeys
    predicate_attr_types = {
        'S_2': {
            "type": "datetime",
            "choices": [str(x) for x in user_log['S_2'].unique()] + ["None"],
        },
        'D_63': {
            "type": "categorical",
            "choices": [str(x) for x in user_log['D_63'].unique()],
        },
        'D_64': {
            "type": "categorical",
            "choices": [str(x) for x in user_log['D_64'].unique()],
        },
        'B_31': {
            "type": "categorical",
            "choices": [str(x) for x in user_log['B_31'].unique()],
        },
    }
    for attr in predicate_attrs:
        if not attr in ['S_2', 'D_63', 'D_64', 'B_31']:
            predicate_attr_types[attr] = {
                "type": "float",
                "low": min(user_log[attr].unique()),
                "high": max(user_log[attr].unique())
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