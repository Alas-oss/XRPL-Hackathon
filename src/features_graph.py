import pandas as pd
from config import ID_COL

def built(train, test):
    allx = pd.concat([train, test], ignore_index=True)
    feats = pd.DataFrame(index=allx[ID_COL])
    # TODO: build account graph from transfers, and degree/
    return feats