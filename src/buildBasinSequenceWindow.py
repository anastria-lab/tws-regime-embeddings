# buildBasinSequenceWindow.py
# Step 7: Build evaluation and continuous analysis windows for basin-level multiscale signals

import os
import pandas as pd
import numpy as np

# Canonical retrospective multiscale product. Both split-contained evaluation
# windows and continuous scientific windows are drawn from the SAME feature product.
ATLAS_INPUT_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"

# Split-aware windows used ONLY for model training / validation / test evaluation.
EVAL_OUTPUT_FILE = "data/processed/window_metadata_v3.parquet"

# Full-record rolling windows used for the hydrological atlas, clustering,
# basin trajectories, and regime-transition analysis.
CONTINUOUS_OUTPUT_FILE = "data/processed/window_metadata_continuous_v3.parquet"

WINDOW_LENGTH = 24
STRIDE = 1

TRAIN_END_YEAR = 2020
VAL_START_YEAR = 2021
VAL_END_YEAR = 2023
TEST_START_YEAR = 2024


def load_dataset(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded dataset: {file_path}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def build_canonical_time(df):
    """Build a canonical monthly timestamp from year/month."""
    df = df.copy()
    df["time"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df


def detect_channel_columns(df):
    """Detect model-ready multiscale channel columns."""
    suffixes = ("_short", "_seasonal", "_long")
    channel_cols = sorted([c for c in df.columns if c.endswith(suffixes)])

    if not channel_cols:
        raise ValueError("No multiscale channel columns found.")

    print("✅ Detected multiscale channels:")
    for c in channel_cols:
        print(f"   - {c}")
    print(f"✅ Total channels: {len(channel_cols)}")
    return channel_cols


def assign_split(df):
    """Assign each basin-month to a time-block split."""
    df = df.copy()

    conditions = [
        df["year"] <= TRAIN_END_YEAR,
        (df["year"] >= VAL_START_YEAR) & (df["year"] <= VAL_END_YEAR),
        df["year"] >= TEST_START_YEAR,
    ]
    choices = ["train", "val", "test"]

    df["split"] = np.select(conditions, choices, default="unknown")

    unknown_count = (df["split"] == "unknown").sum()
    if unknown_count > 0:
        print(f"⚠️ Found {unknown_count} rows with unknown split assignment.")

    return df


def month_diff(t1, t2):
    """Difference in whole months between two timestamps."""
    return (t2.year - t1.year) * 12 + (t2.month - t1.month)


def is_consecutive_month_window(times):
    """Check that timestamps are strictly consecutive monthly timestamps."""
    if len(times) < 2:
        return True

    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    for i in range(len(times) - 1):
        if month_diff(times.iloc[i], times.iloc[i + 1]) != 1:
            return False
    return True


def has_no_missing_channels(window_df, channel_cols):
    """Require finite model channels throughout a window."""
    values = window_df[channel_cols].to_numpy(dtype=float)
    return np.isfinite(values).all()


def make_window_record(window_df, basin_id, channel_cols, window_set, forced_split=None):
    """Create one metadata record from a valid 24-month window."""
    start_split = str(window_df["split"].iloc[0])
    end_split = str(window_df["split"].iloc[-1])
    crosses_boundary = start_split != end_split

    if forced_split is not None:
        split_label = forced_split
    else:
        split_label = start_split if not crosses_boundary else "cross_boundary"

    return {
        "sample_id": None,
        "basin": basin_id,
        "window_set": window_set,
        "split": split_label,
        "start_split": start_split,
        "end_split": end_split,
        "crosses_split_boundary": bool(crosses_boundary),
        "start_time": window_df["time"].iloc[0],
        "end_time": window_df["time"].iloc[-1],
        "start_year": int(window_df["year"].iloc[0]),
        "start_month": int(window_df["month"].iloc[0]),
        "end_year": int(window_df["year"].iloc[-1]),
        "end_month": int(window_df["month"].iloc[-1]),
        "window_length": WINDOW_LENGTH,
        "n_channels": len(channel_cols),
    }


def create_windows_for_group(group_df, basin_id, channel_cols, window_set, forced_split=None):
    """Create rolling windows from one already-selected basin time series."""
    group_df = group_df.sort_values("time").reset_index(drop=True)

    windows = []
    n = len(group_df)

    for start_idx in range(0, n - WINDOW_LENGTH + 1, STRIDE):
        end_idx = start_idx + WINDOW_LENGTH
        window_df = group_df.iloc[start_idx:end_idx]

        if len(window_df) != WINDOW_LENGTH:
            continue
        if not is_consecutive_month_window(window_df["time"]):
            continue
        if not has_no_missing_channels(window_df, channel_cols):
            continue

        windows.append(
            make_window_record(
                window_df=window_df,
                basin_id=basin_id,
                channel_cols=channel_cols,
                window_set=window_set,
                forced_split=forced_split,
            )
        )

    return windows


def finalize_metadata(all_windows, label):
    metadata = pd.DataFrame(all_windows)

    if metadata.empty:
        print(f"⚠️ No {label} windows were created.")
        return metadata

    metadata = metadata.sort_values(["basin", "start_time", "end_time"]).reset_index(drop=True)
    metadata["sample_id"] = np.arange(len(metadata), dtype=np.int64)
    metadata["window_index_within_basin"] = metadata.groupby("basin").cumcount()

    print(f"✅ Built {label} metadata")
    print(f"   Total windows: {len(metadata):,}")
    print(f"   Basins: {metadata['basin'].nunique():,}")
    return metadata


def build_evaluation_window_metadata(df, channel_cols):
    """
    Build split-contained windows for model training/evaluation.

    No window is allowed to cross train/validation/test boundaries.
    """
    all_windows = []
    valid_df = df[df["split"].isin(["train", "val", "test"])].copy()

    for basin_id, basin_df in valid_df.groupby("basin"):
        for split_name, split_df in basin_df.groupby("split"):
            windows = create_windows_for_group(
                split_df,
                basin_id=basin_id,
                channel_cols=channel_cols,
                window_set="evaluation",
                forced_split=split_name,
            )
            all_windows.extend(windows)

    return finalize_metadata(all_windows, "evaluation")


def build_continuous_window_metadata(df, channel_cols):
    """
    Build full-record rolling windows independently of train/val/test boundaries.

    These windows are NOT used to fit the autoencoder. They are encoded only after
    training and are the correct source for the continuous hydrological state-space
    atlas, trajectories, clustering, and transition analysis.
    """
    all_windows = []
    valid_df = df[df["split"].isin(["train", "val", "test"])].copy()

    for basin_id, basin_df in valid_df.groupby("basin"):
        windows = create_windows_for_group(
            basin_df,
            basin_id=basin_id,
            channel_cols=channel_cols,
            window_set="continuous",
            forced_split=None,
        )
        all_windows.extend(windows)

    return finalize_metadata(all_windows, "continuous-analysis")


def sanity_check_window_counts(metadata, label):
    print("\n" + "=" * 70)
    print(f"SANITY CHECK — Window counts ({label})")
    print("=" * 70)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    counts = metadata.groupby("basin").size()
    print(counts.describe())

    if "split" in metadata.columns:
        print("\nCounts by split label:")
        print(metadata["split"].value_counts().sort_index().to_string())


def sanity_check_no_evaluation_split_leakage(metadata):
    """Evaluation metadata must contain no cross-boundary windows."""
    print("\n" + "=" * 70)
    print("SANITY CHECK — Evaluation split containment")
    print("=" * 70)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    n_cross = int(metadata["crosses_split_boundary"].sum())
    if n_cross == 0:
        print("✅ Evaluation windows are fully contained within their assigned split.")
    else:
        raise ValueError(f"Evaluation metadata contains {n_cross} cross-boundary windows.")


def sanity_check_shapes(metadata, expected_channels, label):
    if metadata.empty:
        return

    bad_length = int((metadata["window_length"] != WINDOW_LENGTH).sum())
    bad_channels = int((metadata["n_channels"] != expected_channels).sum())

    if bad_length or bad_channels:
        raise ValueError(
            f"{label}: bad_length={bad_length}, bad_channels={bad_channels}"
        )

    print(f"✅ {label}: all windows are {WINDOW_LENGTH} months × {expected_channels} channels.")


def sanity_check_continuous_start_gaps(metadata):
    """
    Report gaps between successive valid rolling-window starts.

    A gap does not automatically indicate a bug: it can arise when source months or
    model channels are missing. We deliberately do not draw trajectory lines across
    such gaps downstream.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECK — Continuous trajectory start gaps")
    print("=" * 70)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    gap_rows = []
    for basin_id, group in metadata.groupby("basin"):
        times = pd.to_datetime(group.sort_values("start_time")["start_time"]).reset_index(drop=True)
        for i in range(len(times) - 1):
            gap = month_diff(times.iloc[i], times.iloc[i + 1])
            if gap != STRIDE:
                gap_rows.append((basin_id, times.iloc[i], times.iloc[i + 1], gap))

    if not gap_rows:
        print(f"✅ All successive continuous window starts advance by {STRIDE} month.")
    else:
        affected_basins = len(set(r[0] for r in gap_rows))
        print(
            f"⚠️ Found {len(gap_rows):,} gaps in valid window starts across "
            f"{affected_basins:,} basins."
        )
        print("   These will be shown as breaks, not connected jumps, in trajectory plots.")
        print("   We will separately fix/review source-month continuity before the final rerun.")


def save_metadata(metadata, output_file, label):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        metadata.to_parquet(output_file, index=False)
        print(f"✅ Saved {label} metadata: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to save {label} metadata: {e}")
        return None


def main():
    print("--- Step 7: Build evaluation + continuous basin sequence windows ---")
    print(f"Window length: {WINDOW_LENGTH} months | stride: {STRIDE} month")

    atlas_df = load_dataset(ATLAS_INPUT_FILE)
    if atlas_df is None:
        return

    atlas_df = assign_split(build_canonical_time(atlas_df))

    required_cols = {"basin", "year", "month"}
    missing = required_cols - set(atlas_df.columns)
    if missing:
        raise ValueError(f"Atlas dataset is missing required columns: {missing}")

    channel_cols = detect_channel_columns(atlas_df)

    # Both products now use the same retrospective SWT definition. Evaluation
    # windows are still strictly contained inside train/val/test time blocks.
    eval_meta = build_evaluation_window_metadata(atlas_df, channel_cols)
    continuous_meta = build_continuous_window_metadata(atlas_df, channel_cols)

    sanity_check_window_counts(eval_meta, "evaluation")
    sanity_check_no_evaluation_split_leakage(eval_meta)
    sanity_check_shapes(eval_meta, expected_channels=len(channel_cols), label="evaluation")

    sanity_check_window_counts(continuous_meta, "continuous-analysis")
    sanity_check_shapes(continuous_meta, expected_channels=len(channel_cols), label="continuous-analysis")
    sanity_check_continuous_start_gaps(continuous_meta)

    save_metadata(eval_meta, EVAL_OUTPUT_FILE, "evaluation")
    save_metadata(continuous_meta, CONTINUOUS_OUTPUT_FILE, "continuous-analysis")

    print("\n✅ Canonical feature source:")
    print(f"   evaluation windows <- {ATLAS_INPUT_FILE}")
    print(f"   continuous windows <- {ATLAS_INPUT_FILE}")
    print("   train/val/test windows cannot cross time-block boundaries")
    print("   SWT itself is retrospective/non-causal and is NOT a forecasting evaluation")
    print("--- Done ---")


if __name__ == "__main__":
    main()
