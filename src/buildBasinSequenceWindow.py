# buildBasinSequenceWindow.py
# Step 7: Build sequence windows for basin-level multiscale wavelet signals

import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"
OUTPUT_FILE = "data/processed/window_metadata_v3.parquet"

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
    """
    Build a canonical monthly timestamp from year/month.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1)
    )
    return df


def detect_channel_columns(df):
    """
    Detect model-ready multiscale channel columns.
    """
    suffixes = ("_short", "_seasonal", "_long")

    channel_cols = [
        c for c in df.columns
        if c.endswith(suffixes)
    ]

    if not channel_cols:
        raise ValueError("No multiscale channel columns found.")

    channel_cols = sorted(channel_cols)

    print("✅ Detected multiscale channels:")
    for c in channel_cols:
        print(f"   - {c}")

    print(f"✅ Total channels: {len(channel_cols)}")
    return channel_cols


def assign_split(df):
    """
    Assign train/val/test split by time blocks.
    """
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
    """
    Difference in whole months between two timestamps.
    """
    return (t2.year - t1.year) * 12 + (t2.month - t1.month)


def is_consecutive_month_window(times):
    """
    Check that timestamps are strictly consecutive monthly timestamps.
    """
    if len(times) < 2:
        return True

    for i in range(len(times) - 1):
        if month_diff(times.iloc[i], times.iloc[i + 1]) != 1:
            return False
    return True


def has_no_missing_channels(window_df, channel_cols):
    """
    Check no missing values inside the window for selected channels.
    """
    return not window_df[channel_cols].isnull().any().any()


def create_windows_for_group(group_df, basin_id, split_name, channel_cols):
    """
    Create rolling windows for one basin and one split.
    """
    group_df = group_df.sort_values("time").reset_index(drop=True)

    windows = []
    n = len(group_df)

    for start_idx in range(0, n - WINDOW_LENGTH + 1, STRIDE):
        end_idx = start_idx + WINDOW_LENGTH
        window_df = group_df.iloc[start_idx:end_idx]

        # Check exact window length
        if len(window_df) != WINDOW_LENGTH:
            continue

        # Check monthly continuity
        if not is_consecutive_month_window(window_df["time"]):
            continue

        # Check channel completeness
        if not has_no_missing_channels(window_df, channel_cols):
            continue

        row = {
            "sample_id": None,  # fill later
            "basin": basin_id,
            "split": split_name,
            "start_time": window_df["time"].iloc[0],
            "end_time": window_df["time"].iloc[-1],
            "start_year": int(window_df["year"].iloc[0]),
            "start_month": int(window_df["month"].iloc[0]),
            "end_year": int(window_df["year"].iloc[-1]),
            "end_month": int(window_df["month"].iloc[-1]),
            "window_length": WINDOW_LENGTH,
            "n_channels": len(channel_cols),
        }

        windows.append(row)

    return windows


def build_window_metadata(df, channel_cols):
    """
    Build window metadata across all basins and splits.
    """
    all_windows = []

    valid_df = df[df["split"].isin(["train", "val", "test"])].copy()

    for basin_id, basin_df in valid_df.groupby("basin"):
        for split_name, split_df in basin_df.groupby("split"):
            split_df = split_df.sort_values("time").reset_index(drop=True)

            windows = create_windows_for_group(
                split_df,
                basin_id=basin_id,
                split_name=split_name,
                channel_cols=channel_cols
            )
            all_windows.extend(windows)

    metadata = pd.DataFrame(all_windows)

    if metadata.empty:
        print("⚠️ No windows were created.")
        return metadata

    metadata = metadata.reset_index(drop=True)
    metadata["sample_id"] = np.arange(len(metadata))

    print(f"✅ Built window metadata")
    print(f"   Total windows: {len(metadata):,}")

    return metadata


def sanity_check_window_counts(metadata):
    """
    Check window counts per basin and split.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 1 — Window counts per basin")
    print("=" * 60)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    counts = metadata.groupby(["split", "basin"]).size().reset_index(name="n_windows")
    print(counts.groupby("split")["n_windows"].describe())


def sanity_check_no_split_leakage(metadata):
    """
    Check that no basin-time window appears in multiple splits.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 2 — No split leakage")
    print("=" * 60)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    key_counts = (
        metadata.groupby(["basin", "start_time", "end_time"])["split"]
        .nunique()
    )

    leakage = (key_counts > 1).sum()

    if leakage == 0:
        print("✅ No split leakage detected.")
    else:
        print(f"❌ Split leakage detected in {leakage} windows.")


def sanity_check_shapes(metadata, expected_channels):
    """
    Check shape metadata.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 3 — Shape check")
    print("=" * 60)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    bad_length = (metadata["window_length"] != WINDOW_LENGTH).sum()
    bad_channels = (metadata["n_channels"] != expected_channels).sum()

    if bad_length == 0:
        print(f"✅ All windows have length {WINDOW_LENGTH}.")
    else:
        print(f"❌ {bad_length} windows have incorrect length.")

    if bad_channels == 0:
        print(f"✅ All windows have {expected_channels} channels.")
    else:
        print(f"❌ {bad_channels} windows have incorrect channel counts.")


def sanity_check_split_ranges(metadata):
    """
    Check split date ranges.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 4 — Split time ranges")
    print("=" * 60)

    if metadata.empty:
        print("⚠️ Metadata is empty.")
        return

    summary = metadata.groupby("split").agg(
        min_start=("start_time", "min"),
        max_end=("end_time", "max"),
        n_windows=("sample_id", "count"),
    )

    print(summary)


def save_metadata(metadata, output_file):
    """
    Save window metadata to parquet.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        metadata.to_parquet(output_file, index=False)
        print(f"\n✅ Saved window metadata to: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to save window metadata: {e}")
        return None


def main():
    print("--- Build basin sequence windows ---")
    print(f"Window length: {WINDOW_LENGTH} months")
    print(f"Stride: {STRIDE} month")

    df = load_dataset(INPUT_FILE)
    if df is None:
        return

    required_cols = {"basin", "year", "month"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        print(f"❌ Missing required columns: {missing_required}")
        return

    df = build_canonical_time(df)
    df = assign_split(df)

    channel_cols = detect_channel_columns(df)

    metadata = build_window_metadata(df, channel_cols)

    sanity_check_window_counts(metadata)
    sanity_check_no_split_leakage(metadata)
    sanity_check_shapes(metadata, expected_channels=len(channel_cols))
    sanity_check_split_ranges(metadata)

    save_metadata(metadata, OUTPUT_FILE)

    print("--- Done ---")


if __name__ == "__main__":
    main()