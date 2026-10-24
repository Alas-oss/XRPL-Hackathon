# Pumpkin Programmers – XRPL Fraud Detection (Decode & Detect 2026)

We build a model that flags suspicious accounts / activity on the XRP Ledger (XRPL),
combining how accounts behave with how they are connected to each other.

## How it works
1. **Base table** (`src/base_table.py`): turns raw XRPL data into one row per account.
2. **Behavioural features** (`src/features_behav.py`): activity patterns, e.g. how often an account transacts, amounts, timing.
3. **Graph features** (`src/features_graph.py`): who an account sends to / receives from, and how connected it is to other flagged accounts.
4. **Model** (`src/model.py`): learns from these features to score how suspicious each account is.
5. **Pipeline** (`run_pipeline.py`): runs every step end-to-end and writes the predictions.

## How to run
```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Repo layout
| file / folder | what it is | owner |
|---|---|---|
| `src/base_table.py` | builds the account table | Gabriella |
| `src/features_behav.py` | behavioural features | Gabriella |
| `notebooks/` | exploration | Gabriella |
| `src/features_graph.py` | graph features | Aliya |
| `src/model.py` | model training & scoring | Aliya |
| `run_pipeline.py` | end-to-end run | Aliya |
| `TEAM.md`, `README.md` | docs | Lily |

## Results
_To be filled in once the baseline runs (metric, baseline vs final)._

## Disclosure
Libraries, pre-trained models and AI assistants used are listed in the report's disclosure section.

## Licence
MIT – see `LICENSE`.
