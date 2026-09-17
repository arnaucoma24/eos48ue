"""Ensemble inference for eos48ue's four endpoint-clustered chemprop models.

Each cluster is a 10-model chemprop v2 ensemble trained independently (see
`model/framework/fit/`). Prediction shells out to the `chemprop predict` CLI
once per cluster with all 10 `best.pt` paths passed via `--model-paths`,
which chemprop averages into a single ensemble prediction per target.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

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

OUTPUT_COLUMNS = [
    "le_mdck_v2_logpapp",
    "le_mdck_v1_logpapp",
    "caco2_logpapp",
    "mdck_mdr1_loger",
    "logpampa",
    "rat_lm_logclint",
    "human_lm_logclint",
    "mouse_lm_logclint",
    "minipig_lm_logclint",
    "cyno_lm_logclint",
    "dog_lm_logclint",
    "logfu_rat",
    "logfu_human",
    "logfu_mouse",
    "logfu_dog",
    "logfu_monkey",
    "hplc_logfu_hsa",
    "logfu_brain",
    "logfu_mic",
    "nibr_logp",
    "nibr_logd74",
    "cyp3a4_tdi_logkobs",
    "cyp3a4_pic50",
    "cyp2c9_pic50",
    "cyp2d6_pic50",
]


def _canonicalize(smiles_list):
    """Return (original_index, canonical_smiles) only for parseable SMILES."""
    valid_idx, valid_smiles = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            valid_idx.append(i)
            valid_smiles.append(Chem.MolToSmiles(mol))
    return valid_idx, valid_smiles


def _predict_cluster(cluster_name, target_columns, valid_smiles, checkpoints_dir, tmp_dir):
    model_paths = sorted((checkpoints_dir / cluster_name).glob("model_*/best.pt"))
    input_csv = tmp_dir / f"{cluster_name}_input.csv"
    output_csv = tmp_dir / f"{cluster_name}_output.csv"
    pd.DataFrame({"smiles": valid_smiles}).to_csv(input_csv, index=False)

    cmd = [
        "chemprop", "predict",
        "-i", str(input_csv),
        "-o", str(output_csv),
        "--smiles-columns", "smiles",
        "--model-paths", *[str(p) for p in model_paths],
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    df = pd.read_csv(output_csv)
    return df[target_columns].to_numpy(dtype=np.float32)


def predict(smiles_list, checkpoints_dir):
    """Run all four cluster ensembles and return a (n_smiles, 25) float array.

    Rows for unparseable SMILES are filled with NaN; column order matches
    `OUTPUT_COLUMNS`.
    """
    checkpoints_dir = Path(checkpoints_dir)
    n_rows = len(smiles_list)
    outputs = np.full((n_rows, len(OUTPUT_COLUMNS)), np.nan, dtype=np.float32)

    valid_idx, valid_smiles = _canonicalize(smiles_list)
    if not valid_smiles:
        return outputs

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        col_offset = 0
        for cluster_name, target_columns in CLUSTERS.items():
            preds = _predict_cluster(
                cluster_name, target_columns, valid_smiles, checkpoints_dir, tmp_dir
            )
            outputs[np.array(valid_idx), col_offset:col_offset + len(target_columns)] = preds
            col_offset += len(target_columns)

    return outputs
