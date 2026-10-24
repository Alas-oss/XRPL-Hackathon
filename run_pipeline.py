import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from config import ID_COL, LABEL_COL, FEATURE_MODULES, OUT, SEED
from model import cross_validate, precision_at_top
from check_submission import check
import base_table

def main():
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    train, test = base_table.build()

    feats = [train.set_index(ID_COL).drop(columns=[LABEL_COL]),]
    feats_test = [test.set_index(ID_COL)]
    for name in FEATURE_MODULES:
        f = importlib.import_module(name).build(train, test)
        feats.append(f.reindex(train[ID_COL]))
        feats_test.append(f.reindex(test[ID_COL]))
    X = pd.concat(feats, axis=1).select_dtypes(include=[np.number]).fillna(-1)
    X_test = pd.concat(feats_test, axis=1)[X.columns].fillna(-1)
    y = train[LABEL_COL].astype(int).values
    print(f"{X.shape[1]} features, {y.mean():.3%} positives")

    oof, test_scores, model = cross_validate(X.values, y, X_test.values)
    print(f"precision@top1% (OOF): {precision_at_top(y, oof):.3f}")

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("top features:\n", imp.head(15).to_string())

    sub = pd.DataFrame({ID_COL: X_test.index, "score": test_scores})
    path = OUT / "predictions.csv"
    sub.to_csv(path, index=False)
    check(path, expected_rows=len(test))

if __name__ == "__main__":
    main()
