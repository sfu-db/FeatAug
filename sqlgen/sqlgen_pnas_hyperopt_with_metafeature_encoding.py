import copy

import optuna
import time
import duckdb
import numpy as np
from typing import Any, List, Optional, Union, Dict
import warnings

import pandas as pd
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import FrozenTrial
from optuna.trial import create_trial
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from .metafeatures import calculate_all_metafeatures
import logging

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import space_eval

logger = logging.getLogger("03_calculate_metafeatures")
warnings.filterwarnings("ignore")


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
        self.labels = None
        self.query_template = None
        self.optimal_query_list = None
        self.task_type = None

    def reinit(
            self,
    ) -> None:
        self.__init__()

    def build_task(
            self, query_template: QueryTemplate, base_table, labels, relevant_table
    ) -> Any:
        self.query_template = query_template
        self.base_table = base_table
        self.labels = labels
        self.relevant_table = relevant_table
        return self

    def optimize(
            self,
            base_sampler: str = "tpe",
            ml_model: str = "xgb",
            metric: str = "roc_auc",
            direction: str = "maximize",
            outer_budget_type: str = "trial",
            outer_budget: Union[float, int] = 5,
            mi_budget: Union[float, int] = 1000,
            mi_topk: int = 100,
            base_tpe_budget: Union[float, int] = 500,
            turn_on_mi: bool = True,
            turn_on_mapping_func: bool = True,
            train_with_join_keys: bool = False,
            mapping_func: str = "RandomForest",
            seed: int = 0,
            mi_seeds: List = [572, 1567, 2711, 25, 5737, 572, 1567, 2711, 25, 5737],
    ) -> Optional[List]:

        print(mi_seeds)

        self.ml_model = ml_model
        self.metric = metric
        self.optimal_query_list = []
        if metric in ["neg_mean_squared_log_error",
                      "neg_mean_absolute_error",
                      "neg_mean_squared_error",
                      "neg_root_mean_squared_error"]:
            self.task_type = 'regression'
        else:
            self.task_type = 'classification'
        self.train_with_join_keys = train_with_join_keys

        # mi_seeds = [572, 1567, 2711, 25, 5737]
        # mi_seeds = [89, 572, 1024, 25, 3709]
        mi_seeds_pos = 0
        if turn_on_mi:
            # For top-1 mi selection
            # effective_predicate_attrs = []
            # effective_predicate_attrs_type = {}
            # effective_mi_score = 0
            # effective_attr_num = 5
            # for j in range(effective_attr_num):
            #     best_temp_predicate_attrs = []
            #     best_temp_predicate_attrs_type = {}
            #     best_temp_mi_score = 0
            #     seed_list = [0,42,89,1024]
            #     for i in range(len(self.query_template.all_predicate_attrs)):
            #         temp_predicate_attrs = []
            #         for elem in effective_predicate_attrs:
            #             temp_predicate_attrs.append(elem)
            #         if not self.query_template.all_predicate_attrs[i] in temp_predicate_attrs:
            #             temp_predicate_attrs.append(self.query_template.all_predicate_attrs[i])
            #         temp_predicate_attrs_type = {}
            #         for attr in temp_predicate_attrs:
            #             temp_predicate_attrs_type[attr] = self.query_template.all_predicate_attrs_type[attr]
            #         self.query_template.predicate_attrs = temp_predicate_attrs
            #         self.query_template.predicate_attrs_type = temp_predicate_attrs_type
            #
            #         temp_mi_score = 0
            #         for temp_seed in seed_list:
            #             mi_study = optuna.create_study(
            #                 direction="maximize",
            #                 sampler=TPESampler(
            #                     n_startup_trials=20, seed=temp_seed
            #                 ),
            #             )
            #             mi_study.optimize(self._mi_objective_func, n_trials=200)
            #             temp_mi_trials = mi_study.get_trials()
            #             temp_topk_mi_trials = self._rank_trials(temp_mi_trials)[:10]
            #             for elem in temp_topk_mi_trials:
            #                 temp_mi_score += elem["value"]
            #             temp_mi_score /= len(temp_topk_mi_trials)
            #             # best_mi_score = mi_study.best_value
            #             # temp_mi_score += best_mi_score
            #         temp_mi_score = temp_mi_score / len(seed_list)
            #         print(temp_mi_score)
            #         if temp_mi_score > best_temp_mi_score:
            #             best_temp_predicate_attrs = temp_predicate_attrs
            #             best_temp_predicate_attrs_type = temp_predicate_attrs_type
            #             best_temp_mi_score = temp_mi_score
            #         print(best_temp_mi_score)
            #     effective_predicate_attrs = best_temp_predicate_attrs
            #     effective_predicate_attrs_type = best_temp_predicate_attrs_type
            #     effective_mi_score = best_temp_mi_score
            # self.query_template.predicate_attrs = effective_predicate_attrs
            # self.query_template.predicate_attrs_type = effective_predicate_attrs_type

            # PNAS idea with real evaluation (not reward prediction) for each iteration

            effective_predicate_attrs_list = [] # Store current top-K
            previous_layer_predicate_attrs_list = []
            layer_num = 5
            seed_list = [0, 42, 89, 1024]
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
                    best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=200,
                                trials=trials, rstate=np.random.default_rng(temp_seed))
                    temp_best_mi_value = trials.best_trial['result']['loss']
                    # temp_best_mi_value = best_value
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
            previous_layer_predicate_attrs_list = copy.deepcopy(effective_predicate_attrs_list[:1])

            print(len(all_evaluations_current_layer))

            # Train initial predictor (MLP as the first choice)
            X_train = []
            y_train = []
            relevant_table = copy.deepcopy(self.relevant_table)
            for eva in all_evaluations_current_layer:
                sql_attrs = ''
                for attr in eva['predicate_attrs']:
                    sql_attrs += attr + ', '
                feature_sql = (
                    f"SELECT {sql_attrs[:(len(sql_attrs) - 2)]} "
                    f"FROM relevant_table "
                )
                new_relevant_table = duckdb.query(feature_sql).df()
                temp_encoding = []
                mf = calculate_all_metafeatures(new_relevant_table, dataset_name='data', logger=logger)
                for key in mf.keys():
                    if key not in ('PCA', 'Skewnesses', 'Kurtosisses', 'NumSymbols', 'ClassOccurences', 'MissingValues'):
                        temp_encoding.append(float(mf[key].value))
                X_train.append(temp_encoding)
                y_train.append(eva["mi_score"])
                # for attr in eva["predicate_attrs"]:
                #     idx = self.query_template.all_predicate_attrs.index(attr)
                #     temp_encoding[idx] = 1
                #     X_train.append(temp_encoding)
                #     y_train.append(eva["mi_score"])
            clf = RandomForestRegressor(random_state=0)
            clf.fit(X_train, y_train)

            for j in range(1, layer_num):
                explored_sets_current_layer = []
                all_predicted_evaluations_current_layer = []
                all_real_evaluations_current_layer = []
                for k in range(len(previous_layer_predicate_attrs_list)):
                    temp_attr_info = previous_layer_predicate_attrs_list[k]
                    for i in range(len(self.query_template.all_predicate_attrs)):
                        temp_predicate_attrs = copy.deepcopy(temp_attr_info["predicate_attrs"])
                        if not self.query_template.all_predicate_attrs[i] in temp_predicate_attrs:
                            if_explored_set = copy.deepcopy(temp_predicate_attrs)
                            if_explored_set.append(self.query_template.all_predicate_attrs[i])
                            if not set(if_explored_set) in explored_sets_current_layer:
                                temp_predicate_attrs.append(self.query_template.all_predicate_attrs[i])
                                explored_sets_current_layer.append(set(if_explored_set))
                            else:
                                continue
                        else:
                            continue
                        temp_predicate_attrs_type = {}
                        for attr in temp_predicate_attrs:
                            temp_predicate_attrs_type[attr] = self.query_template.all_predicate_attrs_type[attr]

                        sql_attrs = ''
                        for attr in temp_predicate_attrs:
                            sql_attrs += attr + ', '
                        feature_sql = (
                            f"SELECT {sql_attrs[:(len(sql_attrs) - 2)]} "
                            f"FROM relevant_table "
                        )
                        new_relevant_table = duckdb.query(feature_sql).df()

                        temp_preidcate_attrs_encoding = []
                        mf = calculate_all_metafeatures(new_relevant_table, dataset_name='data', logger=logger)
                        for key in mf.keys():
                            if key not in ('PCA', 'Skewnesses', 'Kurtosisses', 'NumSymbols', 'ClassOccurences', 'MissingValues'):
                                temp_preidcate_attrs_encoding.append(float(mf[key].value))

                        temp_mi_score = clf.predict([temp_preidcate_attrs_encoding])
                        # for temp_seed in seed_list:
                        #     mi_study = optuna.create_study(
                        #         direction="maximize",
                        #         sampler=TPESampler(
                        #         n_startup_trials=20, seed=temp_seed
                        #         ),
                        #     )
                        #     mi_study.optimize(self._mi_objective_func, n_trials=200)
                        #     temp_best_mi_value = mi_study.best_value
                        #     temp_mi_score += temp_best_mi_value
                        # temp_mi_score = temp_mi_score / len(seed_list)
                        all_predicted_evaluations_current_layer.append(
                            {"predicate_attrs": temp_predicate_attrs,
                             "predicate_attrs_type": temp_predicate_attrs_type,
                             "mi_score": temp_mi_score}
                        )
                    print(f"The {k}the attr is completed.")
                all_predicted_evaluations_current_layer = sorted(
                    all_predicted_evaluations_current_layer, key=lambda x: x["mi_score"], reverse=False
                )
                selected_attrs = copy.deepcopy(all_predicted_evaluations_current_layer[:1])
                for eva in selected_attrs:
                    self.query_template.predicate_attrs = eva["predicate_attrs"]
                    self.query_template.predicate_attrs_type = eva["predicate_attrs_type"]

                    fspace = self._define_search_space()
                    temp_mi_score = 0
                    for temp_seed in seed_list:
                        trials = Trials()
                        best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=200,
                                    trials=trials, rstate=np.random.default_rng(temp_seed))
                        temp_best_mi_value = trials.best_trial['result']['loss']
                        # temp_best_mi_value = best_value
                        temp_mi_score += temp_best_mi_value
                    temp_mi_score = temp_mi_score / len(seed_list)
                    print(temp_predicate_attrs, temp_mi_score)
                    all_real_evaluations_current_layer.append({
                        "predicate_attrs": eva["predicate_attrs"],
                        "predicate_attrs_type": eva["predicate_attrs_type"],
                        "mi_score": temp_mi_score}
                    )

                # Retrain the predictor (MLP as the first choice)
                for eva in all_real_evaluations_current_layer:
                    sql_attrs = ''
                    for attr in eva["predicate_attrs"]:
                        sql_attrs += attr + ', '
                    feature_sql = (
                        f"SELECT {sql_attrs[:(len(sql_attrs) - 2)]} "
                        f"FROM relevant_table "
                    )
                    new_relevant_table = duckdb.query(feature_sql).df()
                    temp_attrs_encoding = []
                    mf = calculate_all_metafeatures(new_relevant_table, dataset_name='data', logger=logger)
                    for key in mf.keys():
                        if key not in ('PCA', 'Skewnesses', 'Kurtosisses', 'NumSymbols', 'ClassOccurences', 'MissingValues'):
                            temp_attrs_encoding.append(float(mf[key].value))
                    X_train.append(temp_attrs_encoding)
                    y_train.append(eva["mi_score"])
                clf.fit(X_train, y_train)

                effective_predicate_attrs_list += all_real_evaluations_current_layer
                previous_layer_predicate_attrs_list = copy.deepcopy(all_real_evaluations_current_layer)
                print(effective_predicate_attrs_list)

            effective_predicate_attrs_list = sorted(
                effective_predicate_attrs_list, key=lambda x: x["mi_score"], reverse=False
            )
            print(effective_predicate_attrs_list[0])
            self.query_template.predicate_attrs = effective_predicate_attrs_list[0]["predicate_attrs"]
            self.query_template.predicate_attrs_type = effective_predicate_attrs_list[0]["predicate_attrs_type"]
            # self.query_template.predicate_attrs = ['is_auto_renew', 'membership_expire_date']
            # self.query_template.predicate_attrs_type['is_auto_renew'] = self.query_template.all_predicate_attrs_type[
            #     'is_auto_renew']
            # self.query_template.predicate_attrs_type['membership_expire_date'] = \
            #     self.query_template.all_predicate_attrs_type['membership_expire_date']
            #self.query_template.predicate_attrs_type['is_cancel'] = self.query_template.all_predicate_attrs_type['is_cancel']
            fspace = self._define_search_space()

            trials = Trials()
            best = fmin(fn=self._mi_objective_func, space=fspace, algo=tpe.suggest, max_evals=mi_budget,
                        trials=trials, rstate=np.random.default_rng(seed))
            topk_mi_trials = self._rank_trials(trials=trials, search_space=fspace)[:mi_topk]

            # TPE idea search for best combination
            # comb_study = optuna.create_study(
            #     direction="maximize",
            #     sampler=TPESampler(n_startup_trials=20, seed=seed),
            # )
            # comb_study.optimize(self._comb_mi_objective_func, n_trials=1000)

        while outer_budget > 0:
            start = time.time()
            observed_query_list = []
            if turn_on_mi:
                # mi_study = optuna.create_study(
                #     direction="maximize",
                #     sampler=TPESampler(
                #         n_startup_trials=20, seed=mi_seeds[mi_seeds_pos]
                #     ),
                # )
                # mi_study.optimize(self._mi_objective_func, n_trials=mi_budget)
                # mi_seeds_pos += 1
                # # Change for loop according to frozen trials
                # mi_trials = mi_study.get_trials()
                # topk_mi_trials = self._rank_trials(mi_trials)[:mi_topk]
                # Real evaluate topk_mi_trials
                for trial in topk_mi_trials:
                    real_evaluation = self._get_real_evaluation(trial["param"])
                    trial["trial"]["result"]["loss"] = real_evaluation
                    observed_query_list.append(
                        {
                            "param": trial["param"],
                            "trial": trial["trial"],
                            "mi_value": trial["value"],
                            "real_value": real_evaluation,
                        }
                    )

                    # how to warm start with learned mapping function? 需不需要改变inner code？
            # Warm start with mi_study (observed_query_list)
            warm_start_trials = Trials()
            for observed_query in observed_query_list:
                hyperopt_trial = Trials().new_trial_docs(
                    tids=[observed_query["trial"]["tid"]],
                    specs=[observed_query["trial"]["spec"]],
                    results=[observed_query["trial"]["result"]],
                    miscs=[observed_query["trial"]["misc"]]
                )
                warm_start_trials.insert_trial_docs(hyperopt_trial)
                warm_start_trials.refresh()
            best = fmin(fn=self._objective_func, space=fspace, algo=tpe.suggest, max_evals=(base_tpe_budget + mi_topk),
                        trials=warm_start_trials, rstate=np.random.default_rng(seed))
            best_trial = warm_start_trials.best_trial
            vals = self._unpack_values(best_trial)
            real_params = space_eval(fspace, vals)
            self.optimal_query_list.append(
                {"param": real_params, "trial": best_trial, "value": best_trial['result']['loss']})
            end = time.time()

            if outer_budget_type == "trial":
                outer_budget -= 1
            elif outer_budget_type == "time":
                outer_budget -= end - start
            # add new feature to base table
            # arg_list = []
            # for key in self.optimal_query_list[-1]["param"]:
            #     arg_list.append(self.optimal_query_list[-1]["param"][key])
            new_feature, join_keys = self._generate_new_feature(arg_dict=self.optimal_query_list[-1]["param"])
            self.base_table = self.base_table.merge(
                new_feature, how="left", left_on=join_keys, right_on=join_keys
            )

            print(f"Top {outer_budget} is completed!")
        return self.optimal_query_list

    def generate_new_feature(self, arg_list) -> Any:
        new_feature, join_keys = self._generate_new_feature(arg_list)
        return new_feature, join_keys

    def _objective_func(self, params) -> Any:
        # next_trial_param = self._suggest_next_trial(trial)
        new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, copy=False
        )
        new_train_data = new_train_data.fillna(0)
        if not self.train_with_join_keys:
            new_train_data = new_train_data.drop(columns=join_keys)

        # cross-validation score
        if self.ml_model == "xgb":
            if self.task_type == "classification":
                clf = XGBClassifier(random_state=0)
            else:
                clf = XGBRegressor(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.to_frame(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        elif self.ml_model == "rf":
            clf = RandomForestClassifier(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.values.ravel(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        valid_score = scores["test_score"].mean()
        valid_score = -1 * valid_score

        return valid_score

    def _mi_objective_func(self, params):
        new_feature, join_keys = self._generate_new_feature(arg_dict=params)
        df_with_fkeys = self.base_table[self.query_template.fkeys]
        new_feature_after_join = df_with_fkeys.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
        new_feature_after_join = new_feature_after_join.drop(
            columns=self.query_template.fkeys
        )
        new_feature_after_join = new_feature_after_join.fillna(0)

        if self.task_type == 'classification':
            mi_score = mutual_info_classif(
                new_feature_after_join, self.labels, random_state=0
            )
        else:
            mi_score = mutual_info_regression(
                new_feature_after_join, self.labels, random_state=0
            )
        mi_score = -1 * mi_score[0]

        return mi_score

    def _get_real_evaluation(self, param):
        # arg_list = []
        # for key in param:
        #     arg_list.append(param[key])
        new_feature, join_keys = self._generate_new_feature(arg_dict=param)
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, copy=False
        )
        new_train_data = new_train_data.fillna(0)
        if not self.train_with_join_keys:
            new_train_data = new_train_data.drop(columns=join_keys)

        # cross-validation score
        if self.ml_model == "xgb":
            if self.task_type == "classification":
                clf = XGBClassifier(random_state=0)
            else:
                clf = XGBRegressor(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.to_frame(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        elif self.ml_model == "rf":
            clf = RandomForestClassifier(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.values.ravel(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        valid_score = scores["test_score"].mean()
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
                search_space[f"predicate_attr_{predicate_attr}"] = hp.choice(f"predicate_attr_{predicate_attr}",
                                                                             predicate_choices)
            elif predicate_type == "datetime":
                predicate_choices = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["choices"]
                search_space[f"predicate_attr_{predicate_attr}_bound1"] = hp.choice(
                    f"predicate_attr_{predicate_attr}_bound1",
                    predicate_choices)
                search_space[f"predicate_attr_{predicate_attr}_bound2"] = hp.choice(
                    f"predicate_attr_{predicate_attr}_bound2",
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
                search_space[f"groupby_keys_{self.query_template.groupby_keys[0]}"] = hp.choice(
                    f"groupby_keys_{self.query_template.groupby_keys[0]}", [1])
                for groupby_key in self.query_template.groupby_keys[1:]:
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