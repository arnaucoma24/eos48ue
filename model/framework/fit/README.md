# Fit folder — eos48ue (Surrogate ADME Multi-Task Predictor)

No pre-trained checkpoint for this model is publicly available. This folder
contains everything needed to retrain it from the published surrogate
dataset and hyperparameters.

## Background

- **Data source**: Peteani et al. 2024 (Nat. Commun., doi:10.1038/s41467-024-49979-3),
  Supplementary Data 1 — 273,706 public compounds (ZINC, ChEMBL, PROTAC-DB 2.0)
  annotated with synthetic labels for 25 ADME endpoints, predicted by Novartis'
  proprietary internal models. Not committed to the repo (raw CSV is ~137 MB);
  downloaded on demand by `data/00_download_data.py`.
- **Architecture and hyperparameters**: Peteani et al. 2026 (ChemRxiv,
  doi:10.26434/chemrxiv.15005075/v1, Section 2.3) benchmarked several chemprop
  configurations and found one consistently best for surrogate ADME modeling:
  chemprop v2 GNN, 5 message-passing steps, message-passing hidden dim 500, a
  2-hidden-layer FFN of 2000 units, dropout 0, ensemble of 10 models.
- **Endpoint grouping**: the 2026 paper's own "property-specific" clusters only
  cover 21 of 25 endpoints (no cluster for CYP450). We instead use the four
  clusters from the 2024 paper's Supplementary Software, which partition all
  25 endpoints with no gap, retrained with the 2026 paper's hyperparameters
  above (uniformly across all four, superseding the 2024 paper's per-cluster
  hyperparameters):
  - **Permeability** (5): LE-MDCK v1/v2 LogPapp, Caco-2 LogPapp, MDCK-MDR1 LogER, LogPAMPA
  - **Clearance** (6): LogCLint in rat, human, mouse, minipig, cynomolgus monkey, dog liver microsomes
  - **Binding/Lipophilicity** (10): LogFu in rat/human/mouse/dog/monkey, HSA/microsomal/brain binding, LogP, LogD7.4
  - **CYP450** (4): CYP3A4 time-dependent inhibition (logkobs), CYP3A4/CYP2C9/CYP2D6 reversible inhibition (pIC50)

## Steps to reproduce

1. **Download the data** (writes `data/protacdb2.0_zinc_chembl_dataset.csv`):
   ```bash
   python model/framework/fit/data/00_download_data.py
   ```
2. **Clean it** (canonicalizes SMILES with RDKit, drops unparseable rows,
   writes `data/protacdb2.0_zinc_chembl_dataset.clean.csv`):
   ```bash
   python model/framework/fit/src/00_data_cleaning.py
   ```
3. **Train the four cluster ensembles** (requires `chemprop` v2 installed —
   `pip install chemprop`; see hyperparameters above, hardcoded in the script):
   ```bash
   python model/framework/fit/src/01_fit.py
   ```
   This runs `chemprop train` once per cluster, each producing an ensemble of
   10 models under `results/<cluster_name>/`. **Hardware**: the 2024 paper
   trained on NVIDIA V100 GPUs; a GPU is strongly recommended — training all
   four clusters (10-model ensembles, up to ~270k compounds, hidden dim
   500/2000) on CPU is impractical. Expect on the order of hours per cluster
   on a single V100-class GPU; scale down `--epochs`/`--ensemble-size` for a
   quicker smoke test if needed.
4. **Check performance** (aggregates each cluster's held-out test MAE; sanity
   check against the paper's reported CV MAE ~0.03-0.09 and prospective MAE
   ~0.23-0.49, depending on endpoint):
   ```bash
   python model/framework/fit/src/02_performance.py
   ```
5. **Copy the trained ensembles into `model/checkpoints/`**, one subfolder per
   cluster (e.g. `model/checkpoints/permeability/`, `.../clearance/`, etc.),
   then let the ersilia contributor know so `main.py` can be wired up to load
   them. These checkpoint directories are large (10-model ensembles x 4
   clusters) — do not `git add` them; persist via `eosvc` per this repo's
   `CLAUDE.md` before packaging.
