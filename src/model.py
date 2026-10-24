import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from config import LGB_PARAMS, N_FOLDS, SEED

def rank01(x):
    return rankdata(x, method="average") / len(x)

def make_folds(y, groups=None):
    folds = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = k
    return folds

def cross_validate(X, y, X_test, folds=None, params=None):
    params = params or LGB_PARAMS
    folds = make_folds(y) if folds is None else folds
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        pos = max(int(y[tr].sum()), 1)
        model = lgb.LGBMClassifier(**params, scale_pos_weight=(tr.sum() - pos) / pos)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"fold {k}: AP = {average_precision_score(y[va], oof[va]):.4f}")
    print(f"OOF AP = {average_precision_score(y, oof):.4f}  (score = {100*average_precision_score(y, oof):.2f})")
    return oof, rank01(test_pred), model

def precision_at_top(y, scores, frac=0.01):
    n = max(int(len(y) * frac), 1)
    top = np.argsort(-scores)[:n]
    return y[top].mean()
