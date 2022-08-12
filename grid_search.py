## Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Import model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Import metrics
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn import metrics

# Import utils
import tqdm.notebook as tq
from itertools import product
import json
import pandas as pd
import numpy as np
import statistics
from collections import Counter


class GridSearch(object):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.weight = Counter(y_val)[0] / Counter(y_val)[1]

    def ctb(self):
        learning_rate = [0.03, 0.1]
        max_depth = [4, 6, 8, 10]
        num_trees = [50, 250, 500, 1000]
        loss_function = ['Logloss']
        l2_leaf_reg = [3, 5, 10]
        border_count = [254]
        thread_count = [3]

        iterable = list(
            product(learning_rate, max_depth, num_trees, loss_function, l2_leaf_reg, border_count, thread_count))

        dict_param_result = {
            'learning_rate': [],
            'max_depth': [],
            'num_trees': [],
            'loss_function': [],
            'l2_leaf_reg': [],
            'border_count': [],
            'thread_count': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for lr, md, nt, lf, l2, bc, tc in tq.tqdm(iterable, desc="Performing Grid Search"):

            catboost = CatBoostClassifier(learning_rate=lr,
                                          loss_function=lf,
                                          random_seed=42,
                                          eval_metric='AUC',
                                          max_depth=md,
                                          num_trees=nt,
                                          l2_leaf_reg=l2,
                                          border_count=bc,
                                          thread_count=tc,
                                          scale_pos_weight=self.weight,
                                          verbose=0
                                          )
            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                train_data = Pool(data=X_train_kf, label=y_train_kf)
                eval_data = Pool(data=X_val_kf, label=y_val_kf)
                catboost.fit(X=train_data, eval_set=eval_data, plot=False)
                cortes = [90]
                limiares01 = np.percentile(catboost.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (catboost.predict_proba(X_val_kf)[:, 1])
                pred = (catboost.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['learning_rate'].append(lr)
            dict_param_result['max_depth'].append(md)
            dict_param_result['num_trees'].append(nt)
            dict_param_result['loss_function'].append(lf)
            dict_param_result['l2_leaf_reg'].append(l2)
            dict_param_result['border_count'].append(bc)
            dict_param_result['thread_count'].append(tc)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score:
                print("================================================================")
                print(f"Learning Rates: {lr}")
                print(f"Loss Function: {lf}")
                print(f"Max Depth: {md}")
                print(f"Number of Trees: {nt}")
                print(f"l2_leaf_reg {l2}")
                print(f"border_count {bc}")
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_cat = catboost

        max_score = curr_max_score
        best_catboost = curr_best_cat

        return best_catboost, max_score

    def rf(self):

        max_depth = [2, 5, 10, None]
        n_estimators = [10, 20, 50, 100, 200, 500]
        criterion = ['entropy']

        iterable = list(product(max_depth, n_estimators, criterion))

        dict_param_result = {
            'max_depth': [],
            'criterion': [],
            'n_estimators': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for md, ne, c in tq.tqdm(iterable, desc="Performing Grid Search"):

            random_forest = RandomForestClassifier(n_estimators=ne,
                                                   max_depth=md,
                                                   criterion=c,
                                                   random_state=42)

            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                random_forest.fit(X_train_kf, y_train_kf)
                cortes = [90]
                limiares01 = np.percentile(random_forest.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (random_forest.predict_proba(X_val_kf)[:, 1])
                pred = (random_forest.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['max_depth'].append(md)
            dict_param_result['n_estimators'].append(ne)
            dict_param_result['criterion'].append(c)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score:
                print("================================================================")
                print(f"Max Depth: {md}")
                print(f"N Estimators: {ne}")
                print(f"Criterion: {c}")
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_rf = random_forest

        max_score = curr_max_score
        best_rf = curr_best_rf

        return best_rf, max_score

    def knn(self):
        n_neighbors = [6, 8, 10, 12, 14, 16, 18, 20]
        leaf_size = [20, 30, 40]
        p = [1, 2]
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']

        iterable = list(product(n_neighbors, leaf_size, p, weights, metric))

        dict_param_result = {
            'n_neighbors': [],
            'leaf_size': [],
            'p': [],
            'weights': [],
            'metric': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for nn, ls, p, w, m in tq.tqdm(iterable, desc="Performing Grid Search"):

            knn_cv = KNeighborsClassifier(n_neighbors=nn,
                                          weights=w,
                                          leaf_size=ls,
                                          p=p,
                                          metric=m,
                                          n_jobs=-1)

            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                knn_cv.fit(X_train_kf, y_train_kf)
                cortes = [90]
                limiares01 = np.percentile(knn_cv.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (knn_cv.predict_proba(X_val_kf)[:, 1])
                pred = (knn_cv.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['n_neighbors'].append(nn)
            dict_param_result['leaf_size'].append(ls)
            dict_param_result['p'].append(p)
            dict_param_result['weights'].append(w)
            dict_param_result['metric'].append(m)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score:
                print("================================================================")
                print(f"N Neighbors: {nn}")
                print(f"Leaf Size: {ls}")
                print(f"P: {p}")
                print(f"Weights: {w}")
                print(f"Metric {m}")
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_knn = knn_cv

        max_score = curr_max_score
        best_knn = curr_best_knn

        return best_knn, max_score

    def mlp(self):

        hidden_neurons = [20, 50, 100]
        hidden_layers = [2, 3]
        activations = ['logistic', 'tanh', 'relu']
        solvers = ["adam", "sgd"]
        alpha = [0.001, 0.05, 0.01, 0.1]
        batch_sizes = [16, 32, 64]
        learning_rates = [1e-1, 1e-2, 1e-3, 3e-3]
        epochs = [1000, 1500, 2000]

        iterable = list(
            product(hidden_neurons, hidden_layers, activations, solvers, alpha, batch_sizes, learning_rates, epochs))

        dict_param_result = {
            'hidden_neurons': [],
            'hidden_layers': [],
            'activations': [],
            'solvers': [],
            'alpha': [],
            'batch_sizes': [],
            'learning_rates': [],
            'epochs': [],
            'max_iter': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for hn, hl, act, solver, a, bs, lr, epoch in tq.tqdm(iterable, desc="Performing Grid Search"):

            mlp_cv = MLPClassifier(random_state=42,
                                   hidden_layer_sizes=(hn, hl),
                                   learning_rate_init=lr,
                                   batch_size=bs,
                                   alpha=a,
                                   max_iter=epoch,
                                   activation=act,
                                   solver=solver)

            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                mlp_cv.fit(X_train_kf, y_train_kf)
                cortes = [90]
                limiares01 = np.percentile(mlp_cv.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (mlp_cv.predict_proba(X_val_kf)[:, 1])
                pred = (mlp_cv.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['hidden_neurons'].append(hn)
            dict_param_result['hidden_layers'].append(hl)
            dict_param_result['activations'].append(act)
            dict_param_result['solvers'].append(solver)
            dict_param_result['alpha'].append(a)
            dict_param_result['batch_sizes'].append(bs)
            dict_param_result['learning_rates'].append(lr)
            dict_param_result['epochs'].append(epoch)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score and statistics.mean(
                    dict_cross_val['recall']) != 1.0:
                print("================================================================")
                print(f"Hidden Neurons: {hn}")
                print(f"Hidden Layers: {hl}")
                print(f"Activations: {act}")
                print(f"Solvers: {solver}")
                print(f"Batch Sizes: {bs}")
                print(f"Learning Rates: {lr}")
                print(f"Epochs: {epoch}")
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_mlp = mlp_cv

        max_score = curr_max_score
        best_mlp = curr_best_mlp

        return best_mlp, max_score

    def xgb(self):

        min_child_weight = [1, 5, 10]
        gamma = [0.5, 1, 1.5, 2, 5]
        subsample = [0.6, 0.8, 1.0]
        colsample_bytree = [0.6, 0.8, 1.0]
        learning_rate = [0.01, 0.05, 0.1]
        n_estimators = [20, 50, 100, 200]
        max_depth = [3, 4, 5]

        iterable = list(
            product(min_child_weight, gamma, subsample, colsample_bytree, learning_rate, n_estimators, max_depth))

        dict_param_result = {
            'max_depth': [],
            'min_child_weight': [],
            'gamma': [],
            'subsample': [],
            'colsample_bytree': [],
            'n_estimators': [],
            'learning_rate': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for mcw, g, ss, cb, lr, ne, md in tq.tqdm(iterable, desc="Performing Grid Search"):

            xgb_cv = XGBClassifier(learning_rate=lr,
                                   n_estimators=ne,
                                   objective='binary:logistic',
                                   min_child_weight=mcw,
                                   max_depth=md,
                                   gamma=g,
                                   subsample=ss,
                                   colsample_bytree=cb,
                                   scale_pos_weight=self.weight,
                                   nthread=-1)

            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                xgb_cv.fit(X_train_kf, y_train_kf)
                cortes = [90]
                limiares01 = np.percentile(xgb_cv.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (xgb_cv.predict_proba(X_val_kf)[:, 1])
                pred = (xgb_cv.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['max_depth'].append(md)
            dict_param_result['min_child_weight'].append(mcw)
            dict_param_result['gamma'].append(g)
            dict_param_result['subsample'].append(ss)
            dict_param_result['colsample_bytree'].append(cb)
            dict_param_result['n_estimators'].append(ne)
            dict_param_result['learning_rate'].append(lr)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score and statistics.mean(
                    dict_cross_val['recall']) != 1.0:
                print("================================================================")
                print(f"N Estimators: {ne}")
                print(f'Learning Rate: {lr}')
                print(f"Max Depth: {md}")
                print(f'Min Child Weight: {mcw}')
                print(f"Gamma: {g}")
                print(f'Subsample: {ss}')
                print(f"ColSample by Tree: {cb}")
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_xgb = xgb_cv

        max_score = curr_max_score
        best_xgb = curr_best_xgb

        return best_xgb, max_score

    def lgbm(self):

        learning_rate = [0.01, 0.05, 0.1]
        max_depth = [2, 5, 10]
        n_estimators = [20, 50, 100, 200]
        feature_fraction = [0.4, 0.6, 0.8]
        bagging_fraction = [0.6, 0.8, 0.9]

        iterable = list(product(learning_rate, max_depth, n_estimators, feature_fraction, bagging_fraction))

        dict_param_result = {
            'max_depth': [],
            'learning_rate': [],
            'n_estimators': [],
            'feature_fraction': [],
            'bagging_fraction': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for lr, md, ne, ff, bf in tq.tqdm(iterable, desc="Performing Grid Search"):

            lgbm_cv = LGBMClassifier(n_estimators=ne,
                                     learning_rate=lr,
                                     max_depth=md,
                                     feature_fraction=ff,
                                     bagging_fraction=bf,
                                     scale_pos_weight=self.weight,
                                     random_state=42)

            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                lgbm_cv.fit(X_train_kf, y_train_kf)
                cortes = [90]
                limiares01 = np.percentile(lgbm_cv.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (lgbm_cv.predict_proba(X_val_kf)[:, 1])
                pred = (lgbm_cv.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['max_depth'].append(md)
            dict_param_result['learning_rate'].append(lr)
            dict_param_result['n_estimators'].append(ne)
            dict_param_result['feature_fraction'].append(ff)
            dict_param_result['bagging_fraction'].append(bf)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score and statistics.mean(
                    dict_cross_val['recall']) != 1.0:
                print("================================================================")
                print(f"N Estimators: {ne}")
                print(f'Learning Rate: {lr}')
                print(f"Max Depth: {md}")
                print(f'Feature Fraction: {ff}')
                print(f"Bagging Fraction: {bf}")
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_lgbm = lgbm_cv

        max_score = curr_max_score
        best_lgbm = curr_best_lgbm

        return best_lgbm, max_score

    def logreg(self):

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        model = LogisticRegression()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define search space
        space = dict()
        space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
        space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
        space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
        result = search.fit(X, y)

        print('Best Score: %s' % result.best_score_)
        print('Best Hyperparameters: %s' % result.best_params_)

        logreg = result.best_estimator_
        best_score = result.best_score_

        return logreg, best_score

    def ada(self):

        n_estimatores = [10, 20, 50, 100, 200, 500]
        learning_rate = [0.001, 0.01, 0.1]

        iterable = list(product(n_estimatores, learning_rate))

        dict_param_result = {
            'iter': [],
            'n_estimators': [],
            'learning_rate': [],
            'auc': [],
            'recall': [],
            'precision': [],
            'pr_auc': []
        }

        curr_max_score = 0

        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        X = X_train.append(X_val)
        y = y_train.append(y_val)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for ne, lr in tq.tqdm(iterable, desc="Performing Grid Search"):

            ada_cv = AdaBoostClassifier(base_estimator=RandomForestClassifier(),
                                        n_estimators=ne,
                                        learning_rate=lr,
                                        random_state=42)

            dict_cross_val = {
                'auc': [],
                'precision': [],
                'recall': [],
                'pr_auc': []
            }

            skf = StratifiedKFold(n_splits=10)
            for train_index, val_index in skf.split(X, y):
                X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
                y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]
                ada_cv.fit(X_train_kf, y_train_kf)
                cortes = [90]
                limiares01 = np.percentile(ada_cv.predict_proba(X_val_kf)[:, 1], cortes)
                y_pred = (ada_cv.predict_proba(X_val_kf)[:, 1])
                pred = (ada_cv.predict_proba(X_val_kf)[:, 1] >= limiares01[0]).astype(int)
                auc = metrics.roc_auc_score(y_val_kf, y_pred)
                recall = metrics.recall_score(y_val_kf, pred)
                precision = metrics.precision_score(y_val_kf, pred)
                pr_auc = metrics.average_precision_score(y_val_kf, y_pred)

                dict_cross_val['auc'].append(auc)
                dict_cross_val['precision'].append(precision)
                dict_cross_val['recall'].append(recall)
                dict_cross_val['pr_auc'].append(pr_auc)

            dict_param_result['n_estimators'].append(ne)
            dict_param_result['learning_rate'].append(lr)
            dict_param_result['auc'].append(statistics.mean(dict_cross_val['auc']))
            dict_param_result['recall'].append(statistics.mean(dict_cross_val['recall']))
            dict_param_result['precision'].append(statistics.mean(dict_cross_val['precision']))
            dict_param_result['pr_auc'].append(statistics.mean(dict_cross_val['pr_auc']))

            if statistics.mean(dict_cross_val['recall']) > curr_max_score:
                print("================================================================")
                print(f"N Estimators: {ne}")
                print(f'Learning Rate: {lr}')
                print(f'AUC do teste', statistics.mean(dict_cross_val['auc']))
                print(f'Precision do teste', statistics.mean(dict_cross_val['precision']))
                print(f'Recall do teste', statistics.mean(dict_cross_val['recall']))
                print(f'PR-AUC do teste ', statistics.mean(dict_cross_val['pr_auc']))
                print("================================================================")
                curr_max_score = statistics.mean(dict_cross_val['recall'])
                curr_best_ada = ada_cv

        max_score = curr_max_score
        best_ada = curr_best_ada
