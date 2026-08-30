# prepareNOAAONIv6.py
# Validation B1: Download and prepare NOAA CPC Oceanic Niño Index (ONI), ERSSTv6.
#
# Official source:
#   https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
#
# ONI is a 3-month running mean SST anomaly in Niño 3.4. We map each
# overlapping 3-month season to its center month:
#   DJF->Jan, JFM->Feb, ..., NDJ->Dec.
#
# NOAA's historical ENSO episode convention is implemented:
#   El Niño: ONI >= +0.5 °C for at least 5 consecutive overlapping seasons
#   La Niña: ONI <= -0.5 °C for at least 5 consecutive overlapping seasons
#
# Outputs:
#   data/raw/enso/oni_ersstv6_raw.txt
#   data/processed/enso_oni_monthly_v6.csv
#   results/tables/enso_oni_episode_summary.csv

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
RAW_FILE = "data/raw/enso/oni_ersstv6_raw.txt"
MONTHLY_OUTPUT = "data/processed/enso_oni_monthly_v6.csv"
EPISODE_OUTPUT = "results/tables/enso_oni_episode_summary.csv"

SEASON_CENTER_MONTH = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}

WARM_THRESHOLD = 0.5
COLD_THRESHOLD = -0.5
MIN_EPISODE_SEASONS = 5


def download_or_use_existing():
    Path(RAW_FILE).parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"🌐 Downloading NOAA CPC ONI from:\n   {URL}")
        req = urllib.request.Request(
            URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")

        if "SEAS" not in text or "ANOM" not in text:
            raise ValueError("Downloaded content does not look like NOAA ONI data.")

        Path(RAW_FILE).write_text(text, encoding="utf-8")
        print(f"✅ Saved raw ONI file: {RAW_FILE}")
        return RAW_FILE

    except Exception as exc:
        if os.path.exists(RAW_FILE):
            print(f"⚠️ Download failed: {exc}")
            print(f"✅ Using existing local ONI file: {RAW_FILE}")
            return RAW_FILE

        raise RuntimeError(
            "Could not download NOAA ONI and no local fallback exists.\n"
            f"Manually download {URL}\n"
            f"and save it as {RAW_FILE}"
        ) from exc


def parse_oni(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            if parts[0].upper() == "SEAS":
                continue

            season = parts[0].upper()
            if season not in SEASON_CENTER_MONTH:
                continue

            try:
                year = int(parts[1])
                total = float(parts[2])
                anomaly = float(parts[3])
            except ValueError:
                continue

            month = SEASON_CENTER_MONTH[season]
            time = pd.Timestamp(year=year, month=month, day=1)

            rows.append({
                "time": time,
                "season": season,
                "year": year,
                "center_month": month,
                "nino34_total_sst_c": total,
                "oni": anomaly,
            })

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    if df.empty:
        raise ValueError("No ONI rows parsed.")

    if df["time"].duplicated().any():
        dup = df.loc[df["time"].duplicated(keep=False), ["time", "season", "oni"]]
        raise ValueError(f"Duplicate ONI center months found:\n{dup.head(20)}")

    # Require monthly continuity over parsed record.
    expected = pd.date_range(df["time"].min(), df["time"].max(), freq="MS")
    missing = expected.difference(pd.DatetimeIndex(df["time"]))
    if len(missing):
        print(f"⚠️ ONI record has {len(missing)} missing center months.")
        print(f"   First missing: {missing[:10].tolist()}")

    return df


def threshold_phase(oni):
    if oni >= WARM_THRESHOLD:
        return "El_Nino_threshold"
    if oni <= COLD_THRESHOLD:
        return "La_Nina_threshold"
    return "Neutral"


def mark_episode_runs(df):
    df = df.copy()
    df["threshold_phase"] = df["oni"].map(threshold_phase)
    df["episode_phase"] = "Neutral"
    df["episode_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    warm = df["oni"].ge(WARM_THRESHOLD).to_numpy()
    cold = df["oni"].le(COLD_THRESHOLD).to_numpy()

    episode_counter = 0

    for mask, label in [(warm, "El_Nino"), (cold, "La_Nina")]:
        start = 0
        n = len(mask)

        while start < n:
            if not mask[start]:
                start += 1
                continue

            end = start
            while end + 1 < n and mask[end + 1]:
                end += 1

            run_len = end - start + 1
            if run_len >= MIN_EPISODE_SEASONS:
                episode_counter += 1
                df.loc[start:end, "episode_phase"] = label
                df.loc[start:end, "episode_id"] = episode_counter

            start = end + 1

    return df


def build_episode_summary(df):
    active = df[df["episode_phase"] != "Neutral"].copy()
    if active.empty:
        return pd.DataFrame()

    rows = []
    for episode_id, g in active.groupby("episode_id"):
        label = g["episode_phase"].iloc[0]
        rows.append({
            "episode_id": int(episode_id),
            "phase": label,
            "start_center_month": g["time"].min(),
            "end_center_month": g["time"].max(),
            "n_overlapping_seasons": len(g),
            "mean_oni": g["oni"].mean(),
            "min_oni": g["oni"].min(),
            "max_oni": g["oni"].max(),
            "max_abs_oni": g["oni"].abs().max(),
        })

    return pd.DataFrame(rows).sort_values("start_center_month").reset_index(drop=True)


def main():
    print("--- Validation B1: Prepare NOAA CPC ONI (ERSSTv6) ---")

    raw = download_or_use_existing()
    df = parse_oni(raw)
    df = mark_episode_runs(df)
    episodes = build_episode_summary(df)

    os.makedirs(os.path.dirname(MONTHLY_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(EPISODE_OUTPUT), exist_ok=True)

    df.to_csv(MONTHLY_OUTPUT, index=False)
    episodes.to_csv(EPISODE_OUTPUT, index=False)

    grace_era = df[df["time"] >= pd.Timestamp("2002-01-01")].copy()
    phase_counts = grace_era["episode_phase"].value_counts().reindex(
        ["El_Nino", "Neutral", "La_Nina"], fill_value=0
    )

    recent_episodes = episodes[
        episodes["end_center_month"] >= pd.Timestamp("2002-01-01")
    ].copy()

    print(f"✅ Saved monthly ONI: {MONTHLY_OUTPUT}")
    print(f"✅ Saved episode summary: {EPISODE_OUTPUT}")

    print("\nONI coverage:")
    print(f"   First center month: {df['time'].min().date()}")
    print(f"   Last center month:  {df['time'].max().date()}")
    print(f"   Parsed months:      {len(df):,}")

    print("\nGRACE-era official-threshold episode months:")
    for phase, n in phase_counts.items():
        print(f"   {phase:8s}: {int(n):3d}")

    print("\nENSO episodes overlapping 2002-present:")
    if recent_episodes.empty:
        print("   none found")
    else:
        show = recent_episodes[
            [
                "episode_id", "phase", "start_center_month", "end_center_month",
                "n_overlapping_seasons", "min_oni", "max_oni"
            ]
        ]
        print(show.to_string(index=False))

    print("\nDefinition:")
    print("   El Niño / La Niña episode = ONI threshold ±0.5 °C")
    print("   sustained for at least 5 consecutive overlapping 3-month seasons.")
    print("--- Done ---")


if __name__ == "__main__":
    main()
