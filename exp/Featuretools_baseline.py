import pandas as pd
import os
import sys
import warnings
import time
import numpy as np
import featuretools as ft
import random


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_validate

warnings.filterwarnings("ignore")

sys.path.append("../")
sys.path.insert(1, "../exp")

def load_Tmall():
    path = "../exp_data/Tmall/"
    train_data = pd.read_csv(os.path.join(path, "train_format1.csv"))
    test_data = pd.read_csv(os.path.join(path, "test_data.csv"))
    user_log = pd.read_csv(os.path.join(path, "user_log_format1.csv"))
    user_info = pd.read_csv(os.path.join(path, "user_info_format1.csv"))
    user_log = user_log.rename(columns={"seller_id": "merchant_id"})

    random.seed(42)
    sampled_users = random.sample(train_data['user_id'].unique().tolist(), 10000)
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
    test_labels = test_data["label"]
    test_data = test_data.drop(columns=["label"])

    return train_data_sample, train_labels_sample, test_data, test_labels, user_log_sample

def featuretools_Tmall():
    train_data, train_labels, test_data, test_labels, user_log = load_Tmall()
    train_data['user_id_merchant_id'] = train_data['user_id'].map(str) + '_' + train_data['merchant_id'].map(str)
    user_log['user_id_merchant_id'] = user_log['user_id'].map(str) + '_' + user_log['merchant_id'].map(str)
    user_log = user_log.reset_index()

    es = ft.EntitySet(id='tmall')
    es = es.add_dataframe(dataframe_name="train_data",
                      index='user_id_merchant_id',
                      dataframe=train_data)
    es = es.add_dataframe(dataframe_name="user_log",
                      index='index',
                      dataframe=user_log)
    es = es.add_relationship(
        parent_dataframe_name="train_data",
        parent_column_name="user_id_merchant_id",
        child_dataframe_name="user_log",
        child_column_name="user_id_merchant_id"
    )
    fm, features = ft.dfs(entityset=es,
                      target_dataframe_name='train_data',
                      verbose=True)
    fm = fm.fillna(0)

    clf = RandomForestClassifier(random_state=0, class_weight='balanced')
    #clf = XGBClassifier(random_state=0)
    scores = cross_validate(
                clf,
                fm,
                train_labels,
                cv=5,
                scoring='roc_auc',
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
    return scores['test_score'].mean()

if __name__ == "__main__":
    tmall = featuretools_Tmall()
    print(f"Featuretools valid result on the sample of Tmall Dataset: {tmall}")