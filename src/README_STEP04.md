# Step 04 — Leakage-safe preprocessing and explicit atlas/evaluation products

This step fixes the preprocessing leakage identified in the original pipeline while preserving the retrospective scientific atlas.

## 1. `normalizeBasinTimeseries.py`

All basin/month climatologies and basin-specific anomaly standard deviations are now **fit only from years <= 2020**, matching the training-era reference period. Those frozen reference parameters are then applied to the complete record.

Outputs retain the existing filenames:

- `data/processed/basin_dataset_anomaly_normalized.parquet`
- `data/processed/basin_monthly_climatology.parquet`
- `data/processed/basin_anomaly_std.parquet`

New QC:

- `results/tables/normalization_reference_qc.csv`

Interpretation: post-2020 values are anomalies relative to the fixed 2002–2020-style reference period available in the actual data, rather than anomalies whose baseline contains validation/test years.

## 2. `waveletBasinMultiscale.py`

The same normalized record now produces two deliberately different SWT products:

### Scientific / retrospective atlas

`data/processed/basin_dataset_wavelet_multiscale.parquet`

SWT is applied to each basin over the full retrospective record, while respecting the gap policy from Step 03. Use this for:

- continuous UMAP atlas
- K-Means regime discovery
- physical interpretation
- basin trajectories
- regime-transition analysis

### Strict train/validation/test evaluation

`data/processed/basin_dataset_wavelet_multiscale_eval.parquet`

SWT is applied independently inside train, validation, and test blocks. Wavelet support can therefore not cross 2020/2021 or 2023/2024 split boundaries. Use this only for model reconstruction/generalization diagnostics.

## 3. `buildBasinSequenceWindow.py`

- evaluation windows are now built from `basin_dataset_wavelet_multiscale_eval.parquet`
- continuous scientific windows are built from `basin_dataset_wavelet_multiscale.parquet`

The output filenames remain those established in Step 02.

## 4. `trainTemporalAutoencoder.py`

- model fitting + train/validation/test reconstruction use the split-local evaluation SWT dataset
- the model-input scaler is fit on training rows only
- after training, the frozen encoder/scaler are applied to the retrospective atlas SWT dataset to create the continuous scientific embeddings

Outputs remain:

- evaluation embeddings: `data/processed/embeddings_window_level_v3.parquet`
- scientific continuous embeddings: `data/processed/embeddings_window_level_continuous_v3.parquet`

## What this lets you claim

The poster/paper can cleanly distinguish two roles:

1. **Held-out model evaluation:** validation/test years do not define the anomaly baseline, model-input scaling, or cross-boundary SWT features.
2. **Retrospective scientific atlas:** once the representation model is trained, the complete record can be encoded continuously to study evolving basin behaviour.

## Tests performed here

- all four Python files compile successfully
- synthetic test confirmed post-2020 extreme values do not affect the fitted monthly climatology
- synthetic test confirmed evaluation windows cannot cross split boundaries while continuous windows can

PyWavelets (`pywt`) is not installed in the current execution environment, so the actual SWT runtime was not executed here. Run it in your project environment during the final clean rerun.

## Do not rerun the full pipeline yet

Replace these four scripts, then continue to the next scientific gap before doing one clean full rerun.
