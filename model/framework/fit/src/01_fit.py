"""Train the four property-clustered MT-GNN ensembles for eos48ue.

Endpoint grouping follows the four assay clusters defined in Peteani et al.
2024 (Nat. Commun., doi:10.1038/s41467-024-49979-3, Supplementary Software),
which partition all 25 surrogate ADME endpoints with no gap:
Permeability (5), Clearance (6), Binding/Lipophilicity (10), CYP450 (4).

Hyperparameters follow Peteani et al. 2026 (ChemRxiv,
doi:10.26434/chemrxiv.15005075/v1, Section 2.3), which found this single
configuration to outperform the per-cluster hyperparameters used in the 2024
paper: 5 message-passing steps, message-passing hidden dim 500, a 2-hidden-
layer FFN of 2000 units, dropout 0, ensemble of 10.

Run `00_data_cleaning.py` first to produce the cleaned input CSV.

Each cluster is independent and can be submitted as its own SLURM job (see
`slurm/train_<cluster>.sbatch`); pass `--cluster <name>` to train just one,
or omit it to run all four sequentially.
"""

import argparse
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_CSV = DATA_DIR / "protacdb2.0_zinc_chembl_dataset.clean.csv"

CLUSTERS = {
    "permeability": [
        "pred(LE-MDCKv2_LogPapp)",
        "pred(LE-MDCKv1_LogPapp)",
        "pred(Caco-2_LogPapp)",
        "pred(MDCK-MDR1_LogER)",
        "pred(logPAMPA)",
    ],
    "clearance": [
        "pred(rLM LogCLint)",
        "pred(hLM LogCLint)",
        "pred(mLM LogCLint)",
        "pred(minipigLM LogCLint)",
        "pred(cynoLM LogCLint)",
        "pred(dLM LogCLint)",
    ],
    "binding_lipophilicity": [
        "pred(LogFu-Rat)",
        "pred(LogFu-Human)",
        "pred(LogFu-Mouse)",
        "pred(LogFu-Dog)",
        "pred(LogFu-Monkey)",
        "pred(HPLC LogFu HSA)",
        "pred(LogFubrain)",
        "pred(LogFumic)",
        "pred(Direct NIBR LogP)",
        "pred(Direct NIBR LogD7.4)",
    ],
    "cyp450": [
        "pred(logkobs)",
        "pred(CYP3A4_pIC50)",
        "pred(CYP2C9_pIC50)",
        "pred(CYP2D6_pIC50)",
    ],
}

COMMON_ARGS = [
    "--smiles-columns", "smiles",
    "--task-type", "regression",
    "--num-workers", "8",
    "--use-cuikmolmaker-featurization",
    "--split-type", "scaffold_balanced",
    "--split-sizes", "0.85", "0.075", "0.075",
    "--metric", "mae",
    "--depth", "5",
    "--message-hidden-dim", "500",
    "--ffn-hidden-dim", "2000",
    "--ffn-num-layers", "2",
    "--dropout", "0",
    "--ensemble-size", "10",
    "--epochs", "100",
    "--patience", "5",
]


def train_cluster(name, target_columns):
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        ["chemprop", "train", "--data-path", str(DATA_CSV), "--output-dir", str(output_dir)]
        + COMMON_ARGS
        + ["--target-columns", *target_columns]
    )
    print(f"\n=== Training {name} ({len(target_columns)} tasks) ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",
        choices=list(CLUSTERS),
        default=None,
        help="Train only this cluster; omit to run all four sequentially.",
    )
    args = parser.parse_args()

    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run 00_download_data.py and 00_data_cleaning.py first."
        )

    clusters = {args.cluster: CLUSTERS[args.cluster]} if args.cluster else CLUSTERS
    for name, target_columns in clusters.items():
        train_cluster(name, target_columns)


if __name__ == "__main__":
    main()
