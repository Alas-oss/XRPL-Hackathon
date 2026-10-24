from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

ID_COL = "TRANSACTION_HASH"
LABEL_COL = "label"      # TODO: set to the real label column name in the training data
SEED = 42
N_FOLDS = 5

FEATURE_MODULES = [
    # "features_behaviour",  
    # "features_graph",       
]

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
