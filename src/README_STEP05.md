# Step 5 — Continuous regime-transition analysis

## Why this step exists

The abstract promises that basin classifications are tracked across successive 24-month windows to distinguish persistent, gradually changing, and abruptly shifting storage behaviour. The previous pipeline produced window-level cluster labels and trajectory plots, but did not quantify the temporal ordering of those labels.

`analyzeRegimeTransitions.py` turns the continuous window sequence into a quantitative transition analysis.

## Run position

Run this **after** the final versions of:

1. `trainTemporalAutoencoder.py`
2. `latendSpaceVisualization.py` (UMAP is not required by this script, but normally generated here)
3. `clusterEmbeddingRegimes.py`

The required inputs are:

- `data/processed/embeddings_window_level_continuous_v3.parquet`
- `data/processed/window_clusters.parquet`

Then run:

```bash
python analyzeRegimeTransitions.py
```

## What counts as a transition?

A transition is constructed only when two successive windows for the same basin have start dates exactly **one calendar month apart**.

Therefore, if valid windows stop around an observational gap and resume later, the script starts a new continuous segment and **does not connect the two sides**.

This is essential for the GRACE / GRACE-FO gap handling introduced in Step 3.

## Where displacement is measured

K-Means regimes are defined in the 16-D latent space, not in UMAP. For consistency, trajectory displacement is also measured in the **standardized 16-D latent space**.

UMAP remains a visualization only.

## Main outputs

### Window states with gap-aware segment IDs

`data/processed/window_regime_states_with_segments.parquet`

Adds:

- continuous trajectory segment ID
- standardized latent coordinates `zs1 ... zs16`

### Valid adjacent transitions

`data/processed/window_regime_transitions.parquet`

One row per valid one-month step, with:

- basin
- from/to window
- from/to regime
- whether the regime changed
- raw 16-D displacement
- standardized 16-D displacement
- global displacement percentile

### Residence runs

`data/processed/regime_residence_runs.parquet`

Consecutive monthly window-start assignments that remain in the same regime.

### Transition matrices

`results/tables/regime_transitions/regime_transition_counts.csv`

`results/tables/regime_transitions/regime_transition_probabilities.csv`

The second table answers:

> Given regime Ri at one window start, what regime is assigned one month later?

Because adjacent 24-month windows overlap by 23 months, a strong diagonal is expected and should be interpreted as **state-sequence persistence**, not as independence between 24-month events.

### Change-only transition matrix

`results/tables/regime_transitions/regime_transition_probabilities_change_only.csv`

The diagonal is removed before normalization. It answers the more useful question:

> When a basin actually leaves regime Ri, which regime does it enter?

### Regime persistence summary

`results/tables/regime_transitions/regime_persistence_summary.csv`

Contains for each regime:

- outgoing adjacent pairs
- self-transitions
- persistence probability
- number of residence runs
- mean / median / maximum run length

### Basin-level transition metrics

`results/tables/regime_transitions/basin_transition_metrics.csv`

Includes:

- number of continuous segments
- valid adjacent pairs
- number and rate of regime changes
- persistence probability
- dominant regime and fraction
- number of distinct regimes
- normalized regime-occupancy entropy
- mean / median / p95 / maximum adjacent latent displacement
- total latent path length
- net displacement within continuous segments
- path efficiency
- residence-run statistics

These metrics are the quantitative basis for choosing representative basin examples.

## Stable / gradual / abrupt screening

The script creates:

`results/tables/regime_transitions/basin_trajectory_archetype_candidates.csv`

These are deliberately called **candidates**, not validated physical labels.

The default screening is data-driven:

- **abrupt-shift candidate**: at least one adjacent 16-D displacement is in the global top 1%
- **stable candidate**: low regime-change rate and low segment-level net displacement (both in the lower quartile across eligible basins)
- **gradual-migration candidate**: large net displacement and high directional path efficiency, while no individual step exceeds the global 95th-percentile step threshold
- **mixed/dynamic**: does not meet the above screening definitions
- **insufficient-record**: fewer than 12 valid adjacent pairs

Every threshold used in a run is written into both the candidate CSV and:

`results/tables/regime_transitions/transition_analysis_summary.json`

This makes the screening reproducible.

The next climatological-validation step should determine whether the selected candidates correspond to documented droughts, wet extremes, SPEI evolution, ENSO, or other physical drivers before those labels are used as scientific conclusions.

## Figures produced

- `regime_transition_probability_matrix.png`
- `regime_transition_change_only_matrix.png`
- `regime_persistence_probability.png`
- `latent_step_distance_distribution.png`
- `trajectory_archetype_diagnostic.png`

For the final poster, the likely useful results are:

1. one compact transition matrix,
2. a persistence statistic,
3. one stable, one gradual, and one abrupt basin trajectory chosen from the candidate table and then physically validated.

## Important interpretation note

Successive windows are 24 months long with a 1-month stride. Thus adjacent windows share 23 months of observations. Transition probabilities and residence times characterize the **evolution of the learned rolling-window state**, not a Markov chain of statistically independent hydrological events.
