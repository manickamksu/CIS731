# PS 8 interim analysis: tetracycline resistance in food-animal E. coli

Data: FDA NARMS isolate-level data, Food Producing Animals, USDA/FSIS cecal (2013-present),
<https://www.fda.gov/media/93351/download?attachment>, public domain (U.S. government data).
The raw file is kept unmodified in `data/raw/` (not committed); its SHA-256 is recorded in `provenance.csv`.

## Run
```bash
pip install -r interim/requirements.txt
python interim/ps8_narms.py --data data/raw/<narms file> --inspect   # check detected columns
python interim/ps8_narms.py --data data/raw/<narms file>             # full run (a few minutes on CPU)
```

## Outputs
| File | Content |
|---|---|
| `fig1_prevalence_drift.png`, `prevalence_by_year.csv` | Tetracycline resistance by year and animal source, 95% Wilson CIs, FDA GFI #213 (2017) marked |
| `fig2_umap_sweep.png`, `umap_sweep.csv` | UMAP of resistance profiles, n_neighbors x min_dist, random_state = 42, trustworthiness@10 |
| `fig3_coresistance.png`, `coresistance_phi.csv` | Phi correlation between drug resistances |
| `fig4_missingness.png` | Share of isolates tested per drug per year |
| `baselines.csv` | Majority class, logistic regression Scenario B and A; temporal and random splits; advisory |
| `provenance.csv`, `versions.csv`, `metrics.json`, `summary.txt` | Provenance, component versions, counts |

Label: tetracycline R = 1, S = 0, intermediate excluded. Tetracycline-class drugs are never features.
