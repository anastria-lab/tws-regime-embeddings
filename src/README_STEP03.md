# Step 03 — Monthly continuity and GRACE-gap handling

This step fixes the issue that an inner ERA5–GRACE merge can silently remove months before the Stationary Wavelet Transform (SWT).

## Replace these files

- `mergeBasinsTimeseries.py`
- `normalizeBasinTimeseries.py`
- `waveletBasinMultiscale.py`

These versions assume the Level-4 filenames established in Step 01.

## New behaviour

### 1. Complete monthly calendar
`mergeBasinsTimeseries.py` now:

- uses the shared ERA5/GRACE temporal envelope;
- creates an explicit basin × month calendar;
- left-aligns ERA5 and GRACE values onto that calendar;
- keeps missing GRACE months as `NaN` rather than deleting those rows;
- writes a canonical month-start `time` column;
- validates exactly one consecutive row per basin-month;
- saves `results/tables/merged_monthly_continuity_qc.csv`.

### 2. Gap-aware SWT
`waveletBasinMultiscale.py` now:

- verifies the monthly calendar before doing any transform;
- interpolates only **bounded gaps of at most 2 months**;
- leaves longer gaps missing;
- splits each basin-variable record into finite contiguous segments;
- applies SWT independently to segments with at least 24 months;
- never bridges the long GRACE/GRACE-FO mission gap with a synthetic linear interpolation;
- leaves wavelet channels as `NaN` during long gaps, so later 24-month windows cannot cross them;
- records gap/imputation/segment diagnostics for every basin-variable.

### 3. Normalization metadata
`normalizeBasinTimeseries.py` now explicitly treats the canonical `time` column as an ID/time column.

## Scientific consequence

A long observational gap becomes a real break in the learned trajectory. The model will not interpret months on opposite sides of the gap as adjacent observations. Short isolated missing months can still be filled conservatively to avoid losing long otherwise-continuous records.

## Do not rerun the full pipeline yet

Overlay these files after the Step 01 and Step 02 fixes. We will do one clean rerun after the remaining methodological fixes are complete.
