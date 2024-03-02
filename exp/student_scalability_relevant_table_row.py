import pandas as pd
import os
import argparse
import sys
import warnings
import time
import random
import numpy as np
import copy

import featuretools as ft
import argparse

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append("../")
# rom sqlgen.select_proxy_randomly_sqlgen import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_mapping_func import QueryTemplate, SQLGen
# from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
# from sqlgen.dynamic_proxy_sqlgen import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_spearman_proxy import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_pearson_proxy import QueryTemplate, SQLGen
#from sqlgen.sqlgen_random import QueryTemplate, SQLGen
# from sqlgen.sqlgen_with_lr_proxy import QueryTemplate, SQLGen
# from sqlgen.sqlgen_without_query_template_identification import QueryTemplate, SQLGen
from feataug.feataug_scalability_analysis import QueryTemplate, SQLGen

sys.path.insert(1, "../exp")

COL_DUPLICATE_NUM = 0
SAMPLE_COL_NUM = 4


def load_all_data():
    path = "../exp_data/Student_Performance_Game_Play"
    log_all = pd.read_csv(os.path.join(path, "train.csv"))
    train_all = pd.read_csv(os.path.join(path, "train_labels.csv"))
    train_all['session'] = train_all.session_id.apply(lambda x: int(x.split('_')[0]))
    train_all['q'] = train_all.session_id.apply(lambda x: int(x.split('_')[-1][1:]))
    log_all = log_all.rename(columns={"session_id": "session"})

    return log_all, train_all


def load_data(question_number, log_all, train_all, SAMPLE_ROW_NUM_RELEVANT_TABLE: int = 320000):
    # USE THIS TRAIN DATA WITH THESE QUESTIONS
    if question_number <= 3:
        grp = '0-4'
    elif question_number <= 13:
        grp = '5-12'
    elif question_number <= 22:
        grp = '13-22'

    # EXTRACT DATA ACCORDING TO QUESTION NUMBER
    log_question = log_all.loc[log_all.level_group == grp]
    log_question = log_question.drop(columns=['level_group', 'index'])
    all_columns = log_question.columns
    for column in all_columns:
        if log_question[column].isnull().sum() > 0.2 * len(log_question):
            log_question = log_question.drop(columns=[column])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in ['event_name', 'name', 'room_fqid']:
        encoded_labels = le.fit_transform(log_question[column])
        log_question[column] = encoded_labels
    log_question = log_question.fillna(0)
    random.seed(42)
    sampled_session = random.sample(log_question['session'].unique().tolist(), 10000)
    log_question = log_question[log_question['session'].isin(sampled_session)]
    log_question = log_question.sample(n=SAMPLE_ROW_NUM_RELEVANT_TABLE, random_state=0)
    
    train_question = train_all \
        .loc[train_all.q == question_number]
    train_question = train_question.drop(columns=['session_id'])
    train_question = train_question[train_question['session'].isin(sampled_session)]
    label_question = train_question.pop('correct').astype(int)

    X_train, X_rem, y_train, y_rem = train_test_split(train_question, label_question, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    # Original DataFrame
    duplicated_log_question = log_question.copy()
    df_final = duplicated_log_question.copy()

    # Copy and rename columns for the first duplication
    for i in range(COL_DUPLICATE_NUM):
        df1 = duplicated_log_question.copy()
        df1.columns = [f'{col}{i}' for col in duplicated_log_question.columns]

        # Concatenate all DataFrames horizontally
        df_final = pd.concat([df_final, df1], axis=1)
        print(i)
    print(df_final.shape)
    print(len(X_train))

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
    parser.add_argument('-q', '--question_num', type=int, default=1, help = "ML model",required = False)
    parser.add_argument('-rr', '--sampled_row_num', type=int, default=320000, help = "ML model",required = False)
    args = parser.parse_args()
    ml_model = args.ml_model
    question_num = args.question_num
    sampled_row_num = args.sampled_row_num

    seed_list = [0, 42, 89, 550, 572, 1024, 3709, 119, 97]
    log_all, train_all = load_all_data()

    data_number_of_each_question = []
    score_of_each_question = []

    for question_number in [question_num]:
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels, user_log = load_data(question_number, log_all, train_all, SAMPLE_ROW_NUM_RELEVANT_TABLE = sampled_row_num)
        data_number_of_each_question.append(len(train_data))
        test_score_list = []

        fkeys = ["session"]
        agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "ENTROPY", "KURTOSIS", "MODE", "MAD", "MEDIAN"]
        pre_agg_attrs = ['session', 'elapsed_time', 'event_name', 'name', 'level', 'room_coor_x',
                     'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'room_fqid']

        agg_attrs = copy.deepcopy(pre_agg_attrs)
        for i in range(COL_DUPLICATE_NUM):
            agg_attrs += [f'{col}{i}' for col in pre_agg_attrs]
        # random.seed(42)
        # predicate_attrs = random.sample(agg_attrs, SAMPLE_COL_NUM)
        # agg_attrs = [ 'name', 'level', 'room_coor_x']
        #predicate_attrs = agg_attrs
        predicate_attrs = []
        # while len(predicate_attrs) < SAMPLE_COL_NUM:
        #     col = random.choice(agg_attrs)
        for col in agg_attrs:
            if not col in predicate_attrs:
                if not (col.startswith('session0') or \
                        col.startswith('session1') or \
                        col.startswith('elapsed_time0') or \
                        col.startswith('elapsed_time1')):
                    predicate_attrs.append(col)
        print(len(predicate_attrs))
        print(predicate_attrs)
        groupby_keys = fkeys
        predicate_attr_types = {}
        for attr in predicate_attrs:
            if attr == 'elapsed_time':
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
        predicate_attr_types['elapsed_time'] = {
            "type": "float",
            "low": min(user_log['elapsed_time'].unique()),
            "high": max(user_log['elapsed_time'].unique())
        }

        print("Name length:")
        print(len([str(x) for x in user_log['name'].unique()]))
        print("room_coor_x length:")
        print(len([str(x) for x in user_log['room_coor_x'].unique()] + ["None"]))

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

    # avg_score = 0
    # for question_number in range(1, 19):
    #     avg_score += (data_number_of_each_question[question_number] / sum(data_number_of_each_question)) * \
    #                  score_of_each_question[question_number]
    # print(f'Final Avg Score: {avg_score}')
