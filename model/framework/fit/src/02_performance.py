"""Summarize held-out test-set performance for each of the 4 cluster ensembles
trained by `01_fit.py`, as a sanity check against the paper's reported
cross-validation MAE (~0.03-0.09) and prospective MAE (~0.23-0.49) ranges.

Chemprop writes one `test_predictions.csv` per trained model under each
cluster's output directory; this script finds all of them, joins predictions
back to the true surrogate labels by SMILES, and reports per-task MAE.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
CLEAN_CSV = DATA_DIR / "protacdb2.0_zinc_chembl_dataset.clean.csv"


def main():
    truth = pd.read_csv(CLEAN_CSV).set_index("smiles")
    summary_rows = []

    for pred_path in sorted(RESULTS_DIR.glob("*/**/test_predictions.csv")):
        cluster = pred_path.relative_to(RESULTS_DIR).parts[0]
        preds = pd.read_csv(pred_path).set_index("smiles")
        task_columns = [c for c in preds.columns]
        joined = preds.join(truth, how="inner", rsuffix="_true")

        for col in task_columns:
            true_col = f"{col}_true" if f"{col}_true" in joined.columns else col
            if true_col not in joined.columns:
                continue
            mae = (joined[col] - joined[true_col]).abs().mean()
            summary_rows.append(
                {"cluster": cluster, "task": col, "n": len(joined), "mae": mae}
            )

    if not summary_rows:
        print(f"No test_predictions.csv found under {RESULTS_DIR}. Run 01_fit.py first.")
        return

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    out_path = RESULTS_DIR / "performance_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
