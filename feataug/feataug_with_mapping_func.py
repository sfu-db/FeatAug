import copy
import pdb

import optuna
import time
import random
import duckdb
import numpy as np
import tensorflow as tf
from typing import Any, List, Optional, Union, Dict
import warnings
import math

import pandas as pd
from functools import partial
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import FrozenTrial
from optuna.trial import create_trial
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, r_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import space_eval

from deepctr.models import DeepFM, DCNMix
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

warnings.filterwarnings("ignore")

objective_times = 0
real_eval_times = 0


class QueryTemplate(object):
    """Query Template T(F, A, P, K)"""

    def __init__(
            self,
            fkeys,
            agg_funcs,
            agg_attrs,
            all_predicate_attrs,
            predicate_attrs,
            groupby_keys,
            all_predicate_attrs_type,
            predicate_attrs_type,
    ) -> None:
        self.fkeys = fkeys
        self.agg_funcs = agg_funcs
        self.agg_attrs = agg_attrs
        self.all_predicate_attrs = all_predicate_attrs
        self.predicate_attrs = predicate_attrs
        self.groupby_keys = groupby_keys
        self.all_predicate_attrs_type = all_predicate_attrs_type
        self.predicate_attrs_type = predicate_attrs_type


class SQLGen(object):
    def __init__(
            self,
    ) -> None:
        self.train_with_join_keys = None
        self.metric = None
        self.ml_model = None
        self.relevant_table = None
        self.base_table = None
        self.valid_table = None
        self.test_table = None
        self.labels = None
        self.valid_labels = None
        self.test_labels = None
        self.query_template = None
        self.optimal_query_list = None
        self.task_type = None

    def reinit(
            self,
    ) -> None:
        self.__init__()

    def build_task(
            self, query_template: QueryTemplate, base_table, labels, valid_table, valid_labels, test_table, test_labels, relevant_table
    ) -> Any:
        self.query_template = query_template
        self.base_table = base_table
        self.labels = labels
        self.valid_table = valid_table
        self.valid_labels = valid_labels
        self.test_table = test_table
        self.test_labels = test_labels
        self.relevant_table = relevant_table
        return self

    def optimize(
            self,
            base_sampler: str = "tpe",
            ml_model: str = "rf",
            metric: str = "roc_auc",
            direction: str = "maximize",
            outer_budget_type: str = "trial",
            outer_budget: Union[float, int] = 5,
            mi_budget: Union[float, int] = 1000,
            mi_topk: int = 10,
            base_tpe_budget: Union[float, int] = 500,
            turn_on_mi: bool = True,
            turn_on_mapping_func: bool = True,
            train_with_join_keys: bool = False,
            mapping_func: str = "RandomForest",
            seed: int = 0,
            query_template_num: int = 4
    ) -> Optional[List]:

        self.ml_model = ml_model
        self.metric = metric
        self.optimal_query_list = []
        if metric in ["neg_mean_squared_log_error",
                      "neg_mean_absolute_error",
                      "neg_mean_squared_error",
                      "neg_root_mean_squared_error",
                      "root_mean_squared_error"]:
            self.task_type = 'regression'
        else:
            self.task_type = 'classification'
        self.train_with_join_keys = train_with_join_keys

        mi_seeds_pos = 0

        # PNAS idea with real evaluation (not reward prediction) for each iteration

        effective_predicate_attrs_list = [] # Store current top-K
        previous_layer_predicate_attrs_list = []
        layer_num = 3
        seed_list = [0, 42, 89, 1024]
        # 0th layer: no predicate attrs
        self.query_template.predicate_attrs = []
        self.query_template.predicate_attrs_type = {}
        fspace = self._define_search_space()
        temp_mi_score_empty_set = 0
        for temp_seed in seed_list:
            trials = Trials()
            best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=500, trials=trials,
                        rstate=np.random.default_rng(temp_seed),show_progressbar=False)
            temp_best_mi_value = trials.best_trial['result']['loss']
            # temp_best_mi_value = best_value
            temp_mi_score_empty_set += temp_best_mi_value
        temp_mi_score_empty_set = temp_mi_score_empty_set / len(seed_list)
        print(f"MI score without any predicate attrs: {temp_mi_score_empty_set}")
        
        # 1st layer: add all attributes
        all_evaluations_current_layer = []
        for i in range(len(self.query_template.all_predicate_attrs)):
            temp_predicate_attrs = [self.query_template.all_predicate_attrs[i]]
            temp_predicate_attrs_type = {temp_predicate_attrs[0]: self.query_template.all_predicate_attrs_type[temp_predicate_attrs[0]]}
            self.query_template.predicate_attrs = temp_predicate_attrs
            self.query_template.predicate_attrs_type = temp_predicate_attrs_type

            fspace = self._define_search_space()
            temp_mi_score = 0
            for temp_seed in seed_list:
                trials = Trials()
                best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=500,
                            trials=trials, rstate=np.random.default_rng(temp_seed),show_progressbar=False)
                temp_best_mi_value = trials.best_trial['result']['loss']

                temp_mi_score += temp_best_mi_value
            temp_mi_score = temp_mi_score / len(seed_list)
            print(temp_predicate_attrs, temp_mi_score)
            all_evaluations_current_layer.append(
                {"predicate_attrs": temp_predicate_attrs,
                 "predicate_attrs_type": temp_predicate_attrs_type,
                 "mi_score": temp_mi_score}
            )
        all_evaluations_current_layer = sorted(
            all_evaluations_current_layer, key=lambda x: x["mi_score"], reverse=False
        )
        effective_predicate_attrs_list += all_evaluations_current_layer
        previous_layer_predicate_attrs_list = copy.deepcopy(effective_predicate_attrs_list[:2])

        # Train initial predictor (MLP as the first choice)
        X_train = []
        y_train = []
        for eva in all_evaluations_current_layer:
            temp_encoding = np.zeros(len(self.query_template.all_predicate_attrs)).tolist()
            for attr in eva["predicate_attrs"]:
                idx = self.query_template.all_predicate_attrs.index(attr)
                temp_encoding[idx] = 1
                X_train.append(temp_encoding)
                y_train.append(eva["mi_score"])
        clf = DecisionTreeRegressor(random_state=0, max_depth=3)
        clf.fit(X_train, y_train)

        for j in range(1, layer_num):
            explored_sets_current_layer = []
            all_predicted_evaluations_current_layer = []
            all_real_evaluations_current_layer = []
            for k in range(len(previous_layer_predicate_attrs_list)):
                temp_attr_info = previous_layer_predicate_attrs_list[k]
                for i in range(len(self.query_template.all_predicate_attrs)):
                    temp_predicate_attrs = []
                    temp_preidcate_attrs_encoding = np.zeros(len(self.query_template.all_predicate_attrs)).tolist()
                    for elem in temp_attr_info["predicate_attrs"]:
                        temp_predicate_attrs.append(elem)
                        idx = self.query_template.all_predicate_attrs.index(elem)
                        temp_preidcate_attrs_encoding[idx] = 1
                    if not self.query_template.all_predicate_attrs[i] in temp_predicate_attrs:
                        if_explored_set = copy.deepcopy(temp_predicate_attrs)
                        if_explored_set.append(self.query_template.all_predicate_attrs[i])
                        if not set(if_explored_set) in explored_sets_current_layer:
                            temp_predicate_attrs.append(self.query_template.all_predicate_attrs[i])
                            temp_preidcate_attrs_encoding[i] = 1
                            explored_sets_current_layer.append(set(if_explored_set))
                        else:
                            continue
                    else:
                        continue
                    temp_predicate_attrs_type = {}
                    for attr in temp_predicate_attrs:
                        temp_predicate_attrs_type[attr] = self.query_template.all_predicate_attrs_type[attr]
                    self.query_template.predicate_attrs = temp_predicate_attrs
                    self.query_template.predicate_attrs_type = temp_predicate_attrs_type

                    temp_mi_score = clf.predict([temp_preidcate_attrs_encoding])
                    all_predicted_evaluations_current_layer.append(
                        {"predicate_attrs": temp_predicate_attrs,
                         "predicate_attrs_type": temp_predicate_attrs_type,
                         "mi_score": temp_mi_score}
                    )
            all_predicted_evaluations_current_layer = sorted(
                all_predicted_evaluations_current_layer, key=lambda x: x["mi_score"], reverse=False
            )
            selected_attrs = copy.deepcopy(all_predicted_evaluations_current_layer[:2])
            for eva in selected_attrs:
                print(eva["predicate_attrs"])
                self.query_template.predicate_attrs = eva["predicate_attrs"]
                self.query_template.predicate_attrs_type = eva["predicate_attrs_type"]

                fspace = self._define_search_space()
                temp_mi_score = 0
                for temp_seed in seed_list:
                    trials = Trials()
                    best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=500,
                                trials=trials, rstate=np.random.default_rng(temp_seed),show_progressbar=False)
                    temp_best_mi_value = trials.best_trial['result']['loss']
                        # temp_best_mi_value = best_value
                    temp_mi_score += temp_best_mi_value
                temp_mi_score = temp_mi_score / len(seed_list)
                print(eva["predicate_attrs"], temp_mi_score)
                all_real_evaluations_current_layer.append({
                    "predicate_attrs": eva["predicate_attrs"],
                    "predicate_attrs_type": eva["predicate_attrs_type"],
                    "mi_score": temp_mi_score}
                )

            # Retrain the predictor (MLP as the first choice)
            for eva in all_real_evaluations_current_layer:
                temp_attrs_encoding = np.zeros(len(self.query_template.all_predicate_attrs))
                for attr in eva["predicate_attrs"]:
                    idx = self.query_template.all_predicate_attrs.index(attr)
                    temp_attrs_encoding[idx] = 1
                X_train.append(temp_attrs_encoding)
                y_train.append(eva["mi_score"])
            clf.fit(X_train, y_train)

            effective_predicate_attrs_list += all_real_evaluations_current_layer
            previous_layer_predicate_attrs_list = copy.deepcopy(all_real_evaluations_current_layer)
            #print(effective_predicate_attrs_list)

        effective_predicate_attrs_list.append(
            {"predicate_attrs": [],
             "predicate_attrs_type": {},
             "mi_score": temp_mi_score_empty_set}
        )

        effective_predicate_attrs_list = sorted(
            effective_predicate_attrs_list, key=lambda x: x["mi_score"], reverse=False
        )

        topk_query_templates = effective_predicate_attrs_list[:query_template_num]

        for query_template in topk_query_templates:
            print(query_template)
            outer_budget_single_query_template = outer_budget
            self.query_template.predicate_attrs = query_template["predicate_attrs"]
            self.query_template.predicate_attrs_type = query_template["predicate_attrs_type"]

            fspace = self._define_search_space()
            if turn_on_mi and mi_topk > 0:
                trials = Trials()
                best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=mi_budget,
                            trials=trials, rstate=np.random.default_rng(seed), show_progressbar=False)
                ranked_mi_trials = self._rank_trials(trials=trials, search_space=fspace)
                #topk_mi_trials = ranked_mi_trials[:mi_topk]
                random.seed(42)
                topk_mi_trials = random.sample(ranked_mi_trials, mi_topk)

            while outer_budget_single_query_template > 0:

                # Evaluate LR when injecting each feature
                # fspace = self._define_search_space()
                # if turn_on_mi and mi_topk > 0:
                #     trials = Trials()
                #     # best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=mi_budget,
                #     #             trials=trials, rstate=np.random.default_rng(seed), show_progressbar=False)
                #     best = fmin(fn=self._lr_proxy_objective_func, space=fspace, algo=tpe.suggest, max_evals=mi_budget,
                #                  trials=trials, rstate=np.random.default_rng(seed), show_progressbar=False)
                #     topk_mi_trials = self._rank_trials(trials=trials, search_space=fspace)[:mi_topk]

                start = time.time()
                observed_query_list = []
                if turn_on_mi and mi_topk > 0:

                    tmp_mi = []
                    tmp_real_eval = []
                    for trial in topk_mi_trials:
                        real_evaluation = self._get_real_evaluation(trial["param"])
                        tmp_mi.append(trial["value"])
                        tmp_real_eval.append(real_evaluation)

                    regressor = RandomForestRegressor(random_state=0)
                    # from sklearn.linear_model import LinearRegression
                    # regressor = LinearRegression()
                    regressor.fit(np.array(tmp_mi).reshape(-1, 1), np.array(tmp_real_eval).reshape(-1, 1))

                    for trial in ranked_mi_trials:
                        pred_real_evaluation = regressor.predict(np.array([trial["value"]]).reshape(-1, 1))
                        trial["trial"]["result"]["loss"] = pred_real_evaluation
                        observed_query_list.append(
                            {
                                "param": trial["param"],
                                "trial": trial["trial"],
                                "mi_value": trial["value"],
                                "real_value": pred_real_evaluation,
                            }
                        )

                warm_start_trials = Trials()
                for observed_query in observed_query_list:
                    #pdb.set_trace()
                    # hyperopt_trial = Trials().new_trial_docs(
                    #     tids=[observed_query["trial"]["tid"]],
                    #     specs=[observed_query["trial"]["spec"]],
                    #     results=[observed_query["trial"]["result"]],
                    #     miscs=[observed_query["trial"]["misc"]]
                    # )
                    hyperopt_trial = observed_query["trial"]
                    warm_start_trials.insert_trial_doc(hyperopt_trial)
                    warm_start_trials.refresh()
                   
                #pdb.set_trace()
                best = fmin(
                    fn=self._objective_func, 
                    space=fspace, 
                    algo=partial(tpe.suggest, n_startup_jobs=mi_budget), 
                    max_evals=(base_tpe_budget + mi_budget),
                    trials=warm_start_trials, 
                    rstate=np.random.default_rng(seed), 
                    show_progressbar=False
                )
                best_trial = warm_start_trials.best_trial
                vals = self._unpack_values(best_trial)
                real_params = space_eval(fspace, vals)
                self.optimal_query_list.append(
                    {"param": real_params, "trial": best_trial, "value": best_trial['result']['loss']})
                end = time.time()
                if outer_budget_type == "trial":
                    outer_budget_single_query_template -= 1
                elif outer_budget_type == "time":
                    outer_budget_single_query_template -= end - start

                new_feature, join_keys = self._generate_new_feature(arg_dict=self.optimal_query_list[-1]["param"])
                # self.base_table = self.base_table.merge(
                #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe" + str(len(self.optimal_query_list)))
                # )
                # self.valid_table = self.valid_table.merge(
                #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe" + str(len(self.optimal_query_list)))
                # )
                # self.test_table = self.test_table.merge(
                #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe" + str(len(self.optimal_query_list)))
                # )
                self.base_table = self.base_table.merge(
                    new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf")
                )
                self.valid_table = self.valid_table.merge(
                    new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf")
                )
                self.test_table = self.test_table.merge(
                    new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf")
                )

                print(f"Len train cols: {len(self.base_table.columns.values.tolist())}, Train cols: {self.base_table.columns.values.tolist()}")

                train_cols = self.base_table.columns.values.tolist()
                train_to_drop = [x for x in train_cols if x.endswith('_newf')]
                self.base_table = self.base_table.drop(columns=train_to_drop)
                valid_cols = self.valid_table.columns.values.tolist()
                valid_to_drop = [x for x in valid_cols if x.endswith('_newf')]
                self.valid_table = self.valid_table.drop(columns=valid_to_drop)
                test_cols = self.test_table.columns.values.tolist()
                test_to_drop = [x for x in test_cols if x.endswith('_newf')]
                self.test_table = self.test_table.drop(columns=test_to_drop)

                print(f"Top {outer_budget_single_query_template} validation score: {self.optimal_query_list[-1]['value']}!")
                test_score = self._evaluate_test_data()
                print(f"Top {outer_budget_single_query_template} test score: {test_score}!")

        return self.optimal_query_list

    def generate_new_feature(self, arg_dict) -> Any:
        fkeys_in_sql = ""
        for fkey in self.query_template.fkeys:
            fkeys_in_sql += f"{fkey}, "
        fkeys_in_sql = fkeys_in_sql[: (len(fkeys_in_sql) - 2)]

        agg_func_in_sql = self.query_template.agg_funcs[arg_dict['agg_func']]
        agg_attr_in_sql = self.query_template.agg_attrs[arg_dict['agg_attr']]
        # predicate_attrs_label = arg_list[
        #                         2: (len(arg_list) - len(self.query_template.groupby_keys))
        #                         ]
        # groupby_keys_label = arg_list[
        #                      (len(arg_list) - len(self.query_template.groupby_keys)):
        #                      ]

        where_clause_in_sql = ""
        predicate_attrs_label_pos = 0
        for key in arg_dict:
            if key.startswith("predicate_attr"):
                if "bound2" in key:
                    continue
                predicate_attr = key.split("predicate_attr_")[1].split("_bound1")[0]
                predicate_type = self.query_template.all_predicate_attrs_type[predicate_attr][
                    "type"
                ]
                if predicate_type == "categorical":
                    chosen_value = arg_dict[f"predicate_attr_{predicate_attr}"]
                    if chosen_value != "None":
                        where_clause_in_sql += f"{predicate_attr} = {chosen_value} AND "
                    predicate_attrs_label_pos += 1
                elif predicate_type in (
                        "float",
                        "int",
                        "loguniform",
                        "uniform",
                        "discrete_uniform",
                        "datetime",
                ):
                    chosen_value1 = arg_dict[f"predicate_attr_{predicate_attr}_bound1"]
                    chosen_value2 = arg_dict[f"predicate_attr_{predicate_attr}_bound2"]
                    if chosen_value1 == "None" and chosen_value2 != "None":
                        where_clause_in_sql += f"{predicate_attr} >= {chosen_value2} AND "
                    elif chosen_value2 == "None" and chosen_value1 != "None":
                        where_clause_in_sql += f"{predicate_attr} >= {chosen_value1} AND "
                    elif chosen_value1 == "None" and chosen_value2 == "None":
                        continue
                    elif chosen_value1 <= chosen_value2:
                        where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value1} AND {chosen_value2} AND "
                    elif chosen_value2 <= chosen_value1:
                        where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value2} AND {chosen_value1} AND "
                    predicate_attrs_label_pos += 2
                elif predicate_type == "datetime":
                    chosen_value1 = arg_dict[f"predicate_attr_{predicate_attr}_bound1"]
                    chosen_value2 = arg_dict[f"predicate_attr_{predicate_attr}_bound2"]
                    if chosen_value1 == "None" and chosen_value2 != "None":
                        where_clause_in_sql += f"{predicate_attr} >= {chosen_value2} AND "
                    elif chosen_value2 == "None" and chosen_value1 != "None":
                        where_clause_in_sql += f"{predicate_attr} >= {chosen_value1} AND "
                    elif chosen_value1 == "None" and chosen_value2 == "None":
                        continue
                    elif int(chosen_value1) <= int(chosen_value2):
                        where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value1} AND {chosen_value2} AND "
                    elif int(chosen_value2) <= int(chosen_value1):
                        where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value2} AND {chosen_value1} AND "
                    predicate_attrs_label_pos += 2
        where_clause_in_sql = where_clause_in_sql[: (len(where_clause_in_sql) - 5)]

        groupby_clause_in_sql = ""
        join_keys = []
        for i in range(len(self.query_template.groupby_keys)):
            if arg_dict[f"groupby_keys_{self.query_template.groupby_keys[i]}"] == 1:
                groupby_clause_in_sql += self.query_template.groupby_keys[i] + ", "
                join_keys.append(self.query_template.groupby_keys[i])
        groupby_clause_in_sql = groupby_clause_in_sql[
                                : (len(groupby_clause_in_sql) - 2)
                                ]
        fkeys_in_sql = groupby_clause_in_sql

        relevant_table = self.relevant_table
        if len(where_clause_in_sql) > 0:
            feature_sql = (
                f"SELECT {fkeys_in_sql}, {agg_func_in_sql}({agg_attr_in_sql}) "
                f"FROM relevant_table "
                f"WHERE {where_clause_in_sql} "
                f"GROUP BY {groupby_clause_in_sql} "
            )
        else:
            feature_sql = (
                f"SELECT {fkeys_in_sql}, {agg_func_in_sql}({agg_attr_in_sql}) "
                f"FROM relevant_table "
                f"GROUP BY {groupby_clause_in_sql} "
            )

        new_feature = duckdb.query(feature_sql).df()
        # new_feature = new_feature.astype("float")
        # print(new_feature.columns)

        return new_feature, join_keys

    def _objective_func(self, params) -> Any:
        global objective_times
        objective_times += 1
        print(f"Enter objective function {objective_times} times!")
        #next_trial_param = self._suggest_next_trial(trial)
        new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        # new_train_data = self.base_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe_" + str(len(self.optimal_query_list)))
        # )
        # new_valid_data = self.valid_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe_" + str(len(self.optimal_query_list)))
        # )
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        )
        new_valid_data = self.valid_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        )
        new_train_data = new_train_data.fillna(0)
        new_valid_data = new_valid_data.fillna(0)
        if not self.train_with_join_keys:
            new_train_data = new_train_data.drop(columns=self.query_template.fkeys)
            new_valid_data = new_valid_data.drop(columns=self.query_template.fkeys)

        # print(f"new_train_data cols:{new_train_data.columns.values.tolist()}")
        # print(f"new_valid_data cols:{new_valid_data.columns.values.tolist()}")

        new_train_cols = new_train_data.columns.values.tolist()
        new_train_to_drop = [x for x in new_train_cols if x.endswith('_newf')]
        new_train_data = new_train_data.drop(columns=new_train_to_drop)

        new_valid_cols = new_valid_data.columns.values.tolist()
        new_valid_to_drop = [x for x in new_valid_cols if x.endswith('_newf')]
        new_valid_data = new_valid_data.drop(columns=new_valid_to_drop)

        # validation score
        if self.ml_model == "xgb":
            if self.task_type == "classification":
                clf = XGBClassifier(random_state=0, n_jobs=5)
            else:
                clf = XGBRegressor(random_state=0, n_jobs=5)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.to_frame(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.to_frame(), sample_weight=compute_sample_weight("balanced", self.labels.to_frame()))
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.to_frame(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.to_frame(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.to_frame(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model == "rf":
            if self.task_type == "classification":
                clf = RandomForestClassifier(random_state=0, class_weight='balanced')
            else:
                clf = RandomForestRegressor(random_state=0)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.values.ravel())
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model == 'lr':
            if self.task_type == "classification":
                clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
            else:
                clf = LinearRegression()
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.values.ravel())
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model == 'nn':
            if self.task_type == "classification":
                clf = MLPClassifier(random_state=0, alpha=0.01, learning_rate_init=0.01, max_iter=1000)
            else:
                clf = MLPRegressor(random_state=0)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.values.ravel())
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model in ['deepfm', 'dcnv2']:
            all_train = pd.concat([new_train_data, new_valid_data])
            all_train_labels = pd.concat([self.labels, self.valid_labels])
            all_valid = new_valid_data

            # train with only train data
            # all_train = new_train_data
            # all_train_labels = self.labels
            # all_valid = new_valid_data

            tf.random.set_seed(1024)
            sparse_features = []
            dense_features = all_train.columns.values.tolist()

            mms = MinMaxScaler(feature_range=(0,1))
            all_train[dense_features] = mms.fit_transform(all_train[dense_features])
            mms = MinMaxScaler(feature_range=(0,1))
            all_valid[dense_features] = mms.fit_transform(all_valid[dense_features])

            fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=all_train[feat].max() + 1,embedding_dim=4)
                                for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                for feat in dense_features]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

            all_train_dic = {name:all_train[name].values for name in feature_names}
            all_valid_dic = {name:all_valid[name].values for name in feature_names}

            if self.task_type == "classification":
                if self.ml_model == 'deepfm':
                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', seed=1024)
                elif self.ml_model == 'dcnv2':
                    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', seed=1024, dnn_dropout=0.5)
                model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
            else:
                if self.ml_model == 'deepfm':
                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', seed=1024)
                elif self.ml_model == 'dcnv2':
                    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='regression', seed=1024, dnn_dropout=0.5)
                model.compile("adam", "mse", metrics=['mse'], )

            history = model.fit(all_train_dic, all_train_labels.values,
                                batch_size=256, epochs=10, verbose=2, validation_split=0.25, )
            pred_valid = model.predict(all_valid_dic, batch_size=256)

            if self.task_type == "classification":
                score = roc_auc_score(self.valid_labels, pred_valid)
            else:
                score = math.sqrt(mean_squared_error(self.valid_labels, pred_valid))

        valid_score = score
        if self.task_type == "classification":
            valid_score = -1 * valid_score

        return valid_score

    def _mi_objective_func(self, params):

        ##### Take MI as Proxy #####
        new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        fkeys_and_join_keys = list(set(self.query_template.fkeys).union(set(join_keys)))
        df_with_fkeys_and_join_keys = copy.deepcopy(self.base_table[fkeys_and_join_keys])
        new_feature_after_join = df_with_fkeys_and_join_keys.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
        new_feature_after_join = new_feature_after_join.drop(
            columns=fkeys_and_join_keys
        )
        new_feature_after_join = new_feature_after_join.fillna(0)

        # new_feature_after_join_cols = new_feature_after_join.columns.values.tolist()
        # to_drop = [x for x in new_feature_after_join_cols if x.endswith('_newf')]
        # new_feature_after_join = new_feature_after_join.drop(columns=to_drop)

        if self.task_type == 'classification':
            mi_score = mutual_info_classif(
                new_feature_after_join, self.labels, random_state=0
            )
        else:
            # mi_score = mutual_info_regression(
            #     new_feature_after_join, self.labels, random_state=0
            # )
            # pearson correlation
            mi_score = r_regression(new_feature_after_join, self.labels)
        mi_score = -1 * mi_score[0]

        ##### Take LR result as Proxy #####
        # new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        # new_train_data = self.base_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        # )
        # new_valid_data = self.valid_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        # )
        # new_train_data = new_train_data.fillna(0)
        # new_valid_data = new_valid_data.fillna(0)
        # if not self.train_with_join_keys:
        #     new_train_data = new_train_data.drop(columns=self.query_template.fkeys)
        #     new_valid_data = new_valid_data.drop(columns=self.query_template.fkeys)
        # # print(f"new_train_data cols:{new_train_data.columns.values.tolist()}")
        # # print(f"new_valid_data cols:{new_valid_data.columns.values.tolist()}")
        
        # new_train_cols = new_train_data.columns.values.tolist()
        # new_train_to_drop = [x for x in new_train_cols if x.endswith('_newf')]
        # new_train_data = new_train_data.drop(columns=new_train_to_drop)

        # new_valid_cols = new_valid_data.columns.values.tolist()
        # new_valid_to_drop = [x for x in new_valid_cols if x.endswith('_newf')]
        # new_valid_data = new_valid_data.drop(columns=new_valid_to_drop)

        # if self.task_type == "classification":
        #     clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
        # else:
        #     clf = LinearRegression()
        # clf.fit(new_train_data, self.labels.values.ravel())
        # new_valid_pred = clf.predict(new_valid_data)
        # if self.metric == 'roc_auc':
        #     mi_score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
        # elif self.metric == 'f1_macro':
        #     mi_score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
        # elif self.metric == 'root_mean_squared_error':
        #     mi_score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
        #     mi_score = math.sqrt(score)

        # mi_score = -1 * mi_score

        return mi_score

    def _spearman_objective_func(self, params):

        ##### Take Spearman as Proxy #####
        new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        fkeys_and_join_keys = list(set(self.query_template.fkeys).union(set(join_keys)))
        df_with_fkeys_and_join_keys = copy.deepcopy(self.base_table[fkeys_and_join_keys])
        new_feature_after_join = df_with_fkeys_and_join_keys.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
        new_feature_after_join = new_feature_after_join.drop(
            columns=fkeys_and_join_keys
        )
        new_feature_after_join = new_feature_after_join.fillna(0)

        # new_feature_after_join_cols = new_feature_after_join.columns.values.tolist()
        # to_drop = [x for x in new_feature_after_join_cols if x.endswith('_newf')]
        # new_feature_after_join = new_feature_after_join.drop(columns=to_drop)

        # if self.task_type == 'classification':
        #     mi_score = mutual_info_classif(
        #         new_feature_after_join, self.labels, random_state=0
        #     )
        # else:
        #     # mi_score = mutual_info_regression(
        #     #     new_feature_after_join, self.labels, random_state=0
        #     # )
        #     # pearson correlation
        #     mi_score = r_regression(new_feature_after_join, self.labels)
        from scipy import stats
        mi_score = stats.spearmanr(new_feature_after_join, self.labels)
        mi_score = -1 * mi_score.statistic

        ##### Take LR result as Proxy #####
        # new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        # new_train_data = self.base_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        # )
        # new_valid_data = self.valid_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        # )
        # new_train_data = new_train_data.fillna(0)
        # new_valid_data = new_valid_data.fillna(0)
        # if not self.train_with_join_keys:
        #     new_train_data = new_train_data.drop(columns=self.query_template.fkeys)
        #     new_valid_data = new_valid_data.drop(columns=self.query_template.fkeys)
        # # print(f"new_train_data cols:{new_train_data.columns.values.tolist()}")
        # # print(f"new_valid_data cols:{new_valid_data.columns.values.tolist()}")
        
        # new_train_cols = new_train_data.columns.values.tolist()
        # new_train_to_drop = [x for x in new_train_cols if x.endswith('_newf')]
        # new_train_data = new_train_data.drop(columns=new_train_to_drop)

        # new_valid_cols = new_valid_data.columns.values.tolist()
        # new_valid_to_drop = [x for x in new_valid_cols if x.endswith('_newf')]
        # new_valid_data = new_valid_data.drop(columns=new_valid_to_drop)

        # if self.task_type == "classification":
        #     clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
        # else:
        #     clf = LinearRegression()
        # clf.fit(new_train_data, self.labels.values.ravel())
        # new_valid_pred = clf.predict(new_valid_data)
        # if self.metric == 'roc_auc':
        #     mi_score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
        # elif self.metric == 'f1_macro':
        #     mi_score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
        # elif self.metric == 'root_mean_squared_error':
        #     mi_score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
        #     mi_score = math.sqrt(score)

        # mi_score = -1 * mi_score

        return mi_score

    def _lr_proxy_objective_func(self, params):

        ##### Take MI as Proxy #####
        # new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        # fkeys_and_join_keys = list(set(self.query_template.fkeys).union(set(join_keys)))
        # df_with_fkeys_and_join_keys = copy.deepcopy(self.base_table[fkeys_and_join_keys])
        # new_feature_after_join = df_with_fkeys_and_join_keys.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys
        # )
        # new_feature_after_join = new_feature_after_join.drop(
        #     columns=fkeys_and_join_keys
        # )
        # new_feature_after_join = new_feature_after_join.fillna(0)

        # # new_feature_after_join_cols = new_feature_after_join.columns.values.tolist()
        # # to_drop = [x for x in new_feature_after_join_cols if x.endswith('_newf')]
        # # new_feature_after_join = new_feature_after_join.drop(columns=to_drop)

        # if self.task_type == 'classification':
        #     mi_score = mutual_info_classif(
        #         new_feature_after_join, self.labels, random_state=0
        #     )
        # else:
        #     # mi_score = mutual_info_regression(
        #     #     new_feature_after_join, self.labels, random_state=0
        #     # )
        #     # pearson correlation
        #     mi_score = r_regression(new_feature_after_join, self.labels)
        # mi_score = -1 * mi_score[0]

        ##### Take LR result as Proxy #####
        new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        )
        new_valid_data = self.valid_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        )
        new_train_data = new_train_data.fillna(0)
        new_valid_data = new_valid_data.fillna(0)
        if not self.train_with_join_keys:
            new_train_data = new_train_data.drop(columns=self.query_template.fkeys)
            new_valid_data = new_valid_data.drop(columns=self.query_template.fkeys)
        # print(f"new_train_data cols:{new_train_data.columns.values.tolist()}")
        # print(f"new_valid_data cols:{new_valid_data.columns.values.tolist()}")
        
        new_train_cols = new_train_data.columns.values.tolist()
        new_train_to_drop = [x for x in new_train_cols if x.endswith('_newf')]
        new_train_data = new_train_data.drop(columns=new_train_to_drop)

        new_valid_cols = new_valid_data.columns.values.tolist()
        new_valid_to_drop = [x for x in new_valid_cols if x.endswith('_newf')]
        new_valid_data = new_valid_data.drop(columns=new_valid_to_drop)

        if self.task_type == "classification":
            clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
        else:
            clf = LinearRegression()
        clf.fit(new_train_data, self.labels.values.ravel())
        new_valid_pred = clf.predict(new_valid_data)
        if self.metric == 'roc_auc':
            mi_score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
        elif self.metric == 'f1_macro':
            mi_score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
        elif self.metric == 'root_mean_squared_error':
            mi_score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
            mi_score = math.sqrt(score)

        mi_score = -1 * mi_score

        return mi_score

    def _get_real_evaluation(self, param):
        global real_eval_times
        real_eval_times += 1
        print(f"Enter objective function {real_eval_times} times!")
        # arg_list = []
        # for key in param:
        #     arg_list.append(param[key])
        new_feature, join_keys = self._generate_new_feature(arg_dict=param)
        # new_train_data = self.base_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe_" + str(len(self.optimal_query_list)))
        # )
        # new_valid_data = self.valid_table.merge(
        #     new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newfe_" + str(len(self.optimal_query_list)))
        # )
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        )
        new_valid_data = self.valid_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, suffixes=('', "_newf"), copy=False
        )
        new_train_data = new_train_data.fillna(0)
        new_valid_data = new_valid_data.fillna(0)
        if not self.train_with_join_keys:
            new_train_data = new_train_data.drop(columns=self.query_template.fkeys)
            new_valid_data = new_valid_data.drop(columns=self.query_template.fkeys)
        # print(f"new_train_data cols:{new_train_data.columns.values.tolist()}")
        # print(f"new_valid_data cols:{new_valid_data.columns.values.tolist()}")
        
        new_train_cols = new_train_data.columns.values.tolist()
        new_train_to_drop = [x for x in new_train_cols if x.endswith('_newf')]
        new_train_data = new_train_data.drop(columns=new_train_to_drop)

        new_valid_cols = new_valid_data.columns.values.tolist()
        new_valid_to_drop = [x for x in new_valid_cols if x.endswith('_newf')]
        new_valid_data = new_valid_data.drop(columns=new_valid_to_drop)
      
        #new_train, new_valid, new_train_labels, new_valid_labels = train_test_split(new_train_data, self.labels, test_size=0.25, random_state=42)

        # validation score
        if self.ml_model == "xgb":
            if self.task_type == "classification":
                clf = XGBClassifier(random_state=0, n_jobs=5)
            else:
                clf = XGBRegressor(random_state=0, n_jobs=5)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.to_frame(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.to_frame(), sample_weight=compute_sample_weight("balanced", self.labels.to_frame()))
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.to_frame(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.to_frame(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.to_frame(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model == "rf":
            if self.task_type == "classification":
                clf = RandomForestClassifier(random_state=0, class_weight='balanced')
            else:
                clf = RandomForestRegressor(random_state=0)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.values.ravel())
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model == 'lr':
            if self.task_type == "classification":
                clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
            else:
                clf = LinearRegression()
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.values.ravel())
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model == 'nn':
            if self.task_type == "classification":
                clf = MLPClassifier(random_state=0, alpha=0.01, learning_rate_init=0.01, max_iter=1000)
            else:
                clf = MLPRegressor(random_state=0)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(new_train_data, self.labels.values.ravel())
            new_valid_pred = clf.predict(new_valid_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.valid_labels.values.ravel(), new_valid_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.valid_labels.values.ravel(), new_valid_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.valid_labels.values.ravel(), new_valid_pred)
                score = math.sqrt(score)
        elif self.ml_model in ['deepfm', 'dcnv2']:
            all_train = pd.concat([new_train_data, new_valid_data])
            all_train_labels = pd.concat([self.labels, self.valid_labels])
            all_valid = new_valid_data

            # train with only train data
            # all_train = new_train_data
            # all_train_labels = self.labels
            # all_valid = new_valid_data

            tf.random.set_seed(1024)
            sparse_features = []
            dense_features = all_train.columns.values.tolist()

            mms = MinMaxScaler(feature_range=(0,1))
            all_train[dense_features] = mms.fit_transform(all_train[dense_features])
            mms = MinMaxScaler(feature_range=(0,1))
            all_valid[dense_features] = mms.fit_transform(all_valid[dense_features])

            fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=all_train[feat].max() + 1,embedding_dim=4)
                                for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                for feat in dense_features]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

            all_train_dic = {name:all_train[name].values for name in feature_names}
            all_valid_dic = {name:all_valid[name].values for name in feature_names}

            if self.task_type == "classification":
                if self.ml_model == 'deepfm':
                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', seed=1024)
                elif self.ml_model == 'dcnv2':
                    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', seed=1024, dnn_dropout=0.5)
                model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
            else:
                if self.ml_model == 'deepfm':
                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', seed=1024)
                elif self.ml_model == 'dcnv2':
                    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='regression', seed=1024, dnn_dropout=0.5)
                model.compile("adam", "mse", metrics=['mse'], )

            history = model.fit(all_train_dic, all_train_labels.values,
                                batch_size=256, epochs=10, verbose=2, validation_split=0.25, )
            pred_valid = model.predict(all_valid_dic, batch_size=256)

            if self.task_type == "classification":
                score = roc_auc_score(self.valid_labels, pred_valid)
            else:
                score = math.sqrt(mean_squared_error(self.valid_labels, pred_valid))

        valid_score = score
        if self.task_type == "classification":
            valid_score = -1 * valid_score

        return valid_score

    def _define_search_space(self) -> Optional[Dict]:
        search_space = {}
        search_space['agg_func'] = hp.choice('agg_func', [i for i in range(len(self.query_template.agg_funcs))])
        search_space['agg_attr'] = hp.choice('agg_attr', [i for i in range(len(self.query_template.agg_attrs))])

        for predicate_attr in self.query_template.predicate_attrs:
            predicate_type = self.query_template.predicate_attrs_type[predicate_attr][
                "type"
            ]
            if predicate_type == "categorical":
                predicate_choices = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["choices"]
                search_space[f"predicate_attr_{predicate_attr}"] = hp.choice(f"predicate_attr_{predicate_attr}", predicate_choices)
            elif predicate_type == "datetime":
                predicate_choices = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["choices"]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.choice(f"predicate_attr_{predicate_attr}_bound1",
                                                                             predicate_choices)
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.choice(f"predicate_attr_{predicate_attr}_bound2",
                                                                             predicate_choices)
            elif predicate_type == "float":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.uniform(
                    f"predicate_attr_{predicate_attr}_bound1",
                    predicate_low,
                    predicate_high
                )
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.uniform(
                    f"predicate_attr_{predicate_attr}_bound2",
                    predicate_low,
                    predicate_high
                )
            elif predicate_type == "int":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.uniformint(
                    f"predicate_attr_{predicate_attr}_bound1",
                    predicate_low,
                    predicate_high
                )
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.uniformint(
                    f"predicate_attr_{predicate_attr}_bound2",
                    predicate_low,
                    predicate_high
                )
            elif predicate_type == "loguniform":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.loguniform(
                    f"predicate_attr_{predicate_attr}_bound1",
                    predicate_low,
                    predicate_high
                )
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.loguniform(
                    f"predicate_attr_{predicate_attr}_bound2",
                    predicate_low,
                    predicate_high
                )
            elif predicate_type == "uniform":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.uniform(
                    f"predicate_attr_{predicate_attr}_bound1",
                    predicate_low,
                    predicate_high
                )
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.uniform(
                    f"predicate_attr_{predicate_attr}_bound2",
                    predicate_low,
                    predicate_high
                )
            elif predicate_type == "discrete_uniform":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                predicate_q = self.query_template.predicate_attrs_type[predicate_attr][
                    "q"
                ]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.quniform(
                    f"predicate_attr_{predicate_attr}_bound1",
                    predicate_low,
                    predicate_high
                )
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.quniform(
                    f"predicate_attr_{predicate_attr}_bound2",
                    predicate_low,
                    predicate_high
                )

        if len(self.query_template.groupby_keys) == 1:
            search_space[f"groupby_keys_{self.query_template.groupby_keys[0]}"] = hp.choice(
                f"groupby_keys_{self.query_template.groupby_keys[0]}", [1])
        else:
            chosen_groupby_key = random.choice(self.query_template.groupby_keys)
            search_space[f"groupby_keys_{chosen_groupby_key}"] = hp.choice(
                f"groupby_keys_{chosen_groupby_key}", [1])
            #search_space[f"groupby_keys_{self.query_template.groupby_keys[0]}"] = hp.choice(f"groupby_keys_{self.query_template.groupby_keys[0]}", [1])
            for groupby_key in self.query_template.groupby_keys:
                if groupby_key != chosen_groupby_key:
                    search_space[f"groupby_keys_{groupby_key}"] = hp.choice(f"groupby_keys_{groupby_key}", [0, 1])
        return search_space

    def _generate_new_feature(self, arg_dict: Dict = {}):
        fkeys_in_sql = ""
        for fkey in self.query_template.fkeys:
            fkeys_in_sql += f"{fkey}, "
        fkeys_in_sql = fkeys_in_sql[: (len(fkeys_in_sql) - 2)]

        agg_func_in_sql = self.query_template.agg_funcs[arg_dict['agg_func']]
        agg_attr_in_sql = self.query_template.agg_attrs[arg_dict['agg_attr']]
        # predicate_attrs_label = arg_list[
        #                         2: (len(arg_list) - len(self.query_template.groupby_keys))
        #                         ]
        # groupby_keys_label = arg_list[
        #                      (len(arg_list) - len(self.query_template.groupby_keys)):
        #                      ]

        where_clause_in_sql = ""
        predicate_attrs_label_pos = 0
        for i in range(len(self.query_template.predicate_attrs)):
            predicate_attr = self.query_template.predicate_attrs[i]
            predicate_type = self.query_template.predicate_attrs_type[predicate_attr][
                "type"
            ]
            if predicate_type == "categorical":
                chosen_value = arg_dict[f"predicate_attr_{predicate_attr}"]
                if chosen_value != "None":
                    where_clause_in_sql += f"{predicate_attr} = {chosen_value} AND "
                predicate_attrs_label_pos += 1
            elif predicate_type in (
                    "float",
                    "int",
                    "loguniform",
                    "uniform",
                    "discrete_uniform",
                    "datetime",
            ):
                chosen_value1 = arg_dict[f"predicate_attr_{predicate_attr}_bound1"]
                chosen_value2 = arg_dict[f"predicate_attr_{predicate_attr}_bound2"]
                if chosen_value1 == "None" and chosen_value2 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value2} AND "
                elif chosen_value2 == "None" and chosen_value1 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value1} AND "
                elif chosen_value1 == "None" and chosen_value2 == "None":
                    continue
                elif chosen_value1 <= chosen_value2:
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value1} AND {chosen_value2} AND "
                elif chosen_value2 <= chosen_value1:
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value2} AND {chosen_value1} AND "
                predicate_attrs_label_pos += 2
            elif predicate_type == "datetime":
                chosen_value1 = arg_dict[f"predicate_attr_{predicate_attr}_bound1"]
                chosen_value2 = arg_dict[f"predicate_attr_{predicate_attr}_bound2"]
                if chosen_value1 == "None" and chosen_value2 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value2} AND "
                elif chosen_value2 == "None" and chosen_value1 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value1} AND "
                elif chosen_value1 == "None" and chosen_value2 == "None":
                    continue
                elif int(chosen_value1) <= int(chosen_value2):
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value1} AND {chosen_value2} AND "
                elif int(chosen_value2) <= int(chosen_value1):
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value2} AND {chosen_value1} AND "
                predicate_attrs_label_pos += 2
        where_clause_in_sql = where_clause_in_sql[: (len(where_clause_in_sql) - 5)]

        groupby_clause_in_sql = ""
        join_keys = []
        for i in range(len(self.query_template.groupby_keys)):
            if arg_dict[f"groupby_keys_{self.query_template.groupby_keys[i]}"] == 1:
                groupby_clause_in_sql += self.query_template.groupby_keys[i] + ", "
                join_keys.append(self.query_template.groupby_keys[i])
        groupby_clause_in_sql = groupby_clause_in_sql[
                                : (len(groupby_clause_in_sql) - 2)
                                ]
        fkeys_in_sql = groupby_clause_in_sql

        relevant_table = self.relevant_table
        if len(where_clause_in_sql) > 0:
            feature_sql = (
                f"SELECT {fkeys_in_sql}, {agg_func_in_sql}({agg_attr_in_sql}) "
                f"FROM relevant_table "
                f"WHERE {where_clause_in_sql} "
                f"GROUP BY {groupby_clause_in_sql} "
            )
        else:
            feature_sql = (
                f"SELECT {fkeys_in_sql}, {agg_func_in_sql}({agg_attr_in_sql}) "
                f"FROM relevant_table "
                f"GROUP BY {groupby_clause_in_sql} "
            )

        new_feature = duckdb.query(feature_sql).df()
        # new_feature = new_feature.astype("float")
        # print(new_feature.columns)
        new_feature.rename(columns={f"{agg_func_in_sql.lower()}({agg_attr_in_sql})" : str(arg_dict),
                                    f"{agg_func_in_sql.lower()}({agg_attr_in_sql})_x" : str(arg_dict)+"_x",
                                    f"{agg_func_in_sql.lower()}({agg_attr_in_sql})_y" : str(arg_dict)+"_y",
                                    f"{agg_func_in_sql.lower()}(\"{agg_attr_in_sql}\")" : str(arg_dict),
                                    f"{agg_func_in_sql.lower()}(\"{agg_attr_in_sql}\")_x" : str(arg_dict)+"_x",
                                    f"{agg_func_in_sql.lower()}(\"{agg_attr_in_sql}\")_y" : str(arg_dict)+"_y",
                                    f"{agg_func_in_sql.lower()}(\'{agg_attr_in_sql}\')" : str(arg_dict),
                                    f"{agg_func_in_sql.lower()}(\'{agg_attr_in_sql}\')_x" : str(arg_dict)+"_x",
                                    f"{agg_func_in_sql.lower()}(\'{agg_attr_in_sql}\')_y" : str(arg_dict)+"_y",}, inplace=True)

        return new_feature, join_keys

    def _unpack_values(self, trial):
        vals = trial["misc"]["vals"]
        # unpack the one-element lists to values
        # and skip over the 0-element lists
        rval = {}
        for k, v in list(vals.items()):
            if v:
                rval[k] = v[0]
        return rval

    def _rank_trials(self, trials: Any, search_space: Optional[Dict]) -> Optional[List]:
        # extract parameter and values list
        param_value_list = []
        for trial in trials:
            vals = self._unpack_values(trial)
            real_params = space_eval(search_space, vals)
            param_value_list.append({"param": real_params, "trial": trial, "value": trial['result']['loss']})
        param_value_list = sorted(
            param_value_list, key=lambda x: x["value"], reverse=False
        )
        return param_value_list

    def _learn_mapping_func(self, observed_query_list: Optional[List] = None) -> Any:
        X = np.array([x["mi_value"] for x in observed_query_list])
        y = np.array([x["real_value"] for x in observed_query_list])
        clf = RandomForestRegressor(random_state=0)
        # clf = DecisionTreeRegressor(max_depth=2, random_state=0)
        # clf = LinearRegression()
        # clf = MLPRegressor(random_state=0)
        clf.fit(X.reshape(-1, 1), y)
        return clf

    def _output_trials(self, trials):
        params = list(trials[0].params.keys())
        res = {}
        for param in params:
            res[param] = []
        res['target'] = []
        for trial in trials:
            param_values = trial.params
            for param in param_values:
                res[param].append(param_values[param])
            res['target'].append(trial.value)
        res_df = pd.DataFrame(res)
        res_df.to_csv('all_trial_df_1.csv', index=False)
    
    def _evaluate_test_data(self):
        if not self.train_with_join_keys:
            new_train_data = self.base_table.drop(columns=self.query_template.fkeys)
            new_valid_data = self.valid_table.drop(columns=self.query_template.fkeys)
            new_test_data = self.test_table.drop(columns=self.query_template.fkeys)

        new_train_data = new_train_data.fillna(0)
        new_valid_data = new_valid_data.fillna(0)
        new_test_data = new_test_data.fillna(0)

        train_data = pd.concat([new_train_data, new_valid_data])
        train_labels = pd.concat([self.labels, self.valid_labels])
        
        # train_data = new_train_data
        # train_labels = self.labels
        # test score
        if self.ml_model == "xgb":
            if self.task_type == "classification":
                clf = XGBClassifier(random_state=0, n_jobs=5)
            else:
                clf = XGBRegressor(random_state=0, n_jobs=5)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.to_frame(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(train_data, train_labels, sample_weight=compute_sample_weight("balanced", train_labels))
            new_test_pred = clf.predict(new_test_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.test_labels.to_frame(), new_test_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.test_labels.to_frame(), new_test_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.test_labels.to_frame(), new_test_pred)
                score = math.sqrt(score)
        elif self.ml_model == "rf":
            if self.task_type == "classification":
                clf = RandomForestClassifier(random_state=0, class_weight='balanced')
            else:
                clf = RandomForestRegressor(random_state=0)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(train_data, train_labels)
            new_test_pred = clf.predict(new_test_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.test_labels.values.ravel(), new_test_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.test_labels.values.ravel(), new_test_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.test_labels.values.ravel(), new_test_pred)
                score = math.sqrt(score)
        elif self.ml_model == 'lr':
            if self.task_type == "classification":
                clf = LogisticRegression(random_state = 0, penalty='l2', class_weight='balanced')
            else:
                clf = LinearRegression()
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(train_data, train_labels)
            new_test_pred = clf.predict(new_test_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.test_labels.values.ravel(), new_test_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.test_labels.values.ravel(), new_test_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.test_labels.values.ravel(), new_test_pred)
                score = math.sqrt(score)
        elif self.ml_model == 'nn':
            if self.task_type == "classification":
                clf = MLPClassifier(random_state=0, alpha=0.01, learning_rate_init=0.01, max_iter=1000)
            else:
                clf = MLPRegressor(random_state=0)
            # scores = cross_validate(
            #     clf,
            #     new_train_data,
            #     self.labels.values.ravel(),
            #     cv=5,
            #     scoring=self.metric,
            #     return_train_score=True,
            #     n_jobs=-1,
            #     return_estimator=True,
            # )
            clf.fit(train_data, train_labels)
            new_test_pred = clf.predict(new_test_data)
            if self.metric == 'roc_auc':
                score = roc_auc_score(self.test_labels.values.ravel(), new_test_pred)
            elif self.metric == 'f1_macro':
                score = f1_score(self.test_labels.values.ravel(), new_test_pred, average='macro')
            elif self.metric == 'root_mean_squared_error':
                score = mean_squared_error(self.test_labels.values.ravel(), new_test_pred)
                score = math.sqrt(score)
        elif self.ml_model in ['deepfm', 'dcnv2']:
            all_train = pd.concat([new_train_data, new_valid_data])
            all_train_labels = pd.concat([self.labels, self.valid_labels])
            all_test = new_test_data

            tf.random.set_seed(1024)
            sparse_features = []
            dense_features = all_train.columns.values.tolist()

            mms = MinMaxScaler(feature_range=(0,1))
            all_train[dense_features] = mms.fit_transform(all_train[dense_features])
            mms = MinMaxScaler(feature_range=(0,1))
            all_test[dense_features] = mms.fit_transform(all_test[dense_features])

            fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=all_train[feat].max() + 1,embedding_dim=4)
                                for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                for feat in dense_features]

            dnn_feature_columns = fixlen_feature_columns
            linear_feature_columns = fixlen_feature_columns

            feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

            all_train_dic = {name:all_train[name].values for name in feature_names}
            all_test_dic = {name:all_test[name].values for name in feature_names}

            if self.task_type == "classification":
                if self.ml_model == 'deepfm':
                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', seed=1024)
                elif self.ml_model == 'dcnv2':
                    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', seed=1024, dnn_dropout=0.5)
                model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
            else:
                if self.ml_model == 'deepfm':
                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', seed=1024)
                elif self.ml_model == 'dcnv2':
                    model = DCNMix(linear_feature_columns, dnn_feature_columns, task='regression', seed=1024, dnn_dropout=0.5)
                model.compile("adam", "mse", metrics=['mse'], )

            history = model.fit(all_train_dic, all_train_labels.values,
                                batch_size=256, epochs=10, verbose=2, validation_split=0.25, )
            pred_test = model.predict(all_test_dic, batch_size=256)

            if self.task_type == "classification":
                score = roc_auc_score(self.test_labels, pred_test)
            else:
                score = math.sqrt(mean_squared_error(self.test_labels, pred_test))

        return score
    
    
    
    
        

        
                


