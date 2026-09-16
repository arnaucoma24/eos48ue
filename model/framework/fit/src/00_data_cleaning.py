"""Standardize SMILES in the surrogate ADME dataset before training.

Reads `model/framework/fit/data/protacdb2.0_zinc_chembl_dataset.csv`
(produced by `data/00_download_data.py`), canonicalizes each SMILES with
RDKit, and drops rows that fail to parse. Writes a new file rather than
modifying the original, per this repo's fit/README.md convention.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV = DATA_DIR / "protacdb2.0_zinc_chembl_dataset.csv"
OUTPUT_CSV = DATA_DIR / "protacdb2.0_zinc_chembl_dataset.clean.csv"


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def main():
    df = pd.read_csv(INPUT_CSV)
    df["smiles"] = df["smiles"].apply(canonicalize)
    n_before = len(df)
    df = df.dropna(subset=["smiles"])
    n_after = len(df)
    print(f"Dropped {n_before - n_after} unparseable SMILES ({n_after} remaining)")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
