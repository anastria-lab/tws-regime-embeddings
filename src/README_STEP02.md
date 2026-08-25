# Step 02 — Continuous rolling-window analysis fix

This update separates two different purposes that were previously mixed together.

## A. Evaluation windows

Output:

`data/processed/window_metadata_v3.parquet`

These windows are built separately inside train / validation / test periods. They are used only to train and evaluate the temporal autoencoder. No evaluation window crosses a split boundary.

## B. Continuous scientific-analysis windows

Output:

`data/processed/window_metadata_continuous_v3.parquet`

These are 24-month, stride-1 rolling windows generated over each basin's full valid monthly record without stopping at train / validation / test boundaries. Windows crossing those boundaries are retained and explicitly marked.

After the autoencoder is trained on the evaluation windows, the same trained encoder is applied to the continuous windows and writes:

`data/processed/embeddings_window_level_continuous_v3.parquet`

This file is now the canonical source for:

- the UMAP hydrological state-space atlas,
- KMeans regime discovery,
- basin trajectories,
- later regime-transition analysis.

The previous evaluation embedding file remains:

`data/processed/embeddings_window_level_v3.parquet`

and is retained for model diagnostics only.

## Important trajectory change

Trajectory plots no longer connect across missing valid-window starts. If successive valid 24-month windows do not start exactly one month apart, the plotted line is broken. This prevents artificial long straight jumps across data gaps.

## Updated scripts

1. `buildBasinSequenceWindow.py`
2. `trainTemporalAutoencoder.py`
3. `latendSpaceVisualization.py`
4. `clusterEmbeddingRegimes.py`

## Do not rerun the full pipeline yet

A later cleanup step will address monthly continuity / GRACE gaps before SWT and preprocessing leakage. Replace these scripts now, but wait for the final clean-run sequence before regenerating all results.
