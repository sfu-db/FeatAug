import pandas as pd
import os
import sys
import warnings
import time
import random
import numpy as np
import argparse

import featuretools as ft
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append("../")
from sqlgen.sqlgen_pnas_hyperopt_onehot_encoding_multi_query_template import QueryTemplate, SQLGen
sys.path.insert(1, "../exp")


def load_data():
    train = pd.read_csv("../exp_data/Household/train.csv")

    # Groupby the household and figure out the number of unique values
    all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

    # Iterate through each household
    for household in not_equal.index:
        # Find the correct label (for the head of household)
        true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

        # Set the correct label for all members in the household
        train.loc[train['idhogar'] == household, 'Target'] = true_target

    # Groupby the household and figure out the number of unique values
    all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

    train_valid = train.loc[train['parentesco1'] == 1, ['idhogar', 'Target']].copy()

    mapping = {"yes": 1, "no": 0}

    # Fill in the values with the correct mapping
    train['dependency'] = train['dependency'].replace(mapping).astype(np.float64)
    train['edjefa'] = train['edjefa'].replace(mapping).astype(np.float64)
    train['edjefe'] = train['edjefe'].replace(mapping).astype(np.float64)

    train['v18q1'] = train['v18q1'].fillna(0)
    train.loc[(train['tipovivi1'] == 1), 'v2a1'] = 0
    train['rez_esc'] = train['rez_esc'].fillna(0)

    train = train[[x for x in train if not x.startswith('SQB')]]
    train = train.drop(columns=['agesq'])

    for column in train.columns:
        train[column] = train[column].fillna(0)

    hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo',
               'paredpreb', 'pisocemento', 'pareddes', 'paredmad',
               'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother',
               'pisonatur', 'pisonotiene', 'pisomadera',
               'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
               'abastaguadentro', 'abastaguafuera', 'abastaguano',
               'public', 'planpri', 'noelec', 'coopele', 'sanitario1',
               'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',
               'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
               'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
               'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
               'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3',
               'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',
               'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
               'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

    hh_ordered = ['rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2',
                  'r4t3', 'v18q1', 'tamhog', 'tamviv', 'hhsize', 'hogar_nin',
                  'hogar_adul', 'hogar_mayor', 'hogar_total', 'bedrooms', 'qmobilephone']

    hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']

    ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
                'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
                'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5',
                'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10',
                'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3',
                'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
                'instlevel9', 'mobilephone']

    es = ft.EntitySet(id='households')
    es.add_dataframe(dataframe_name='ind',
                     dataframe=train,
                     index='Id')

    es.normalize_dataframe(base_dataframe_name='ind',
                           new_dataframe_name='household',
                           index='idhogar',
                           additional_columns= ['Target'])

    identity_df = es.dataframe_dict['ind']
    household_df = es.dataframe_dict['household']

    X = household_df
    label = X.pop('Target') - 1

    X_train, X_rem, y_train, y_rem = train_test_split(X, label, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, identity_df

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
    if 'idhogar' in train_data.columns.tolist():
        train_data = train_data.drop(columns=['idhogar'])
    if 'idhogar' in test_data.columns.tolist():
        test_data = test_data.drop(columns=['idhogar'])
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
    score = f1_score(test_labels, predictions, average='macro')

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--ml_model', type=str, default='rf', help = "ML model",required = False)
    args = parser.parse_args()
    ml_model = args.ml_model

    train_data, train_labels, valid_data, valid_labels, test_data, test_labels, user_log = load_data()
    seed_list = [0, 42, 89, 550, 572, 1024, 3709]
    test_score_list = []

    fkeys = ["idhogar"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG", "APPROX_COUNT_DISTINCT", "VAR_POP", "STDDEV_POP"]
    agg_attrs = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo',
               'paredpreb', 'pisocemento', 'pareddes', 'paredmad',
               'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother',
               'pisonatur', 'pisonotiene', 'pisomadera',
               'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
               'abastaguadentro', 'abastaguafuera', 'abastaguano',
               'public', 'planpri', 'noelec', 'coopele', 'sanitario1',
               'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',
               'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
               'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
               'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
               'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3',
               'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',
               'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
               'lugar4', 'lugar5', 'lugar6', 'area1', 'area2'] + ['rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2',
                  'r4t3', 'v18q1', 'tamhog', 'tamviv', 'hhsize', 'hogar_nin',
                  'hogar_adul', 'hogar_mayor', 'hogar_total', 'bedrooms', 'qmobilephone'] + \
                ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding'] + ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
                'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
                'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5',
                'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10',
                'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3',
                'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
                'instlevel9', 'mobilephone']
    random.seed(0)
    predicate_attrs = random.sample(agg_attrs, 20)
    print(predicate_attrs)

    groupby_keys = fkeys
    predicate_attr_types = {}
    for attr in predicate_attrs:
        predicate_attr_types[attr] = {
            "type": "categorical",
            "choices": [str(x) for x in user_log[attr].unique()],
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
            metric="f1_macro",
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