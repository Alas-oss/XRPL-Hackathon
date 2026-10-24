import pandas as pd
from config import ID_COL

def build(test, train):
    allx = pd.concat([train, test], ignore_index=True)
    feats = pd.DataFrame(index=alls[ID_COL])
    # TODO: round-amount flags, just-under-threshold, burstiness, etc.
    return feats