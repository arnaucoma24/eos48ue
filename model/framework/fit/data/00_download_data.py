"""Download the surrogate ADME dataset (Peteani et al. 2024, Nat. Commun.,
doi:10.1038/s41467-024-49979-3) used to train this model.

The dataset is released as Supplementary Data 1 of the paper: 273,706 public
compounds (ZINC, ChEMBL, PROTAC-DB 2.0) annotated with synthetic labels for 25
ADME endpoints, predicted by Novartis' proprietary internal models. It is
downloaded here rather than committed to the repo because the raw CSV is
~137 MB.
"""

import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent
ZIP_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-024-49979-3/MediaObjects/"
    "41467_2024_49979_MOESM4_ESM.zip"
)
ZIP_PATH = DATA_DIR / "surrogate_data.zip"
CSV_NAME = "protacdb2.0_zinc_chembl_dataset.csv"


def main():
    print(f"Downloading surrogate dataset from {ZIP_URL} ...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(ZIP_PATH, "wb") as out_file:
        out_file.write(response.read())

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        member = next(n for n in zf.namelist() if n.endswith(CSV_NAME))
        with zf.open(member) as src, open(DATA_DIR / CSV_NAME, "wb") as dst:
            dst.write(src.read())

    ZIP_PATH.unlink()
    print(f"Saved to {DATA_DIR / CSV_NAME}")


if __name__ == "__main__":
    main()
