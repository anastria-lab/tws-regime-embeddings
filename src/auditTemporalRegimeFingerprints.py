# auditTemporalRegimeFingerprints.py
#
# Diagnostic question:
#   Do the current latent-space clusters represent repeatable MODES OF
#   HYDROLOGICAL EVOLUTION over the 24-month windows?
#
# This audit deliberately does NOT ask whether each state always means
# "wet / intermediate / dry".
#
# It uses the exact multiscale channels supplied to the autoencoder and the
# exact continuous 24-month windows. It evaluates:
#
#   1) calendar-month dependence (are clusters merely seasonal phase?);
#   2) basin-balanced 24-month temporal templates;
#   3) cross-basin template coherence;
#   4) simple dynamic descriptors such as TWS-long slope/change, seasonal
#      amplitude, short-term variability, precipitation/storage evolution;
#   5) current raw k=2 versus final raw k=3.
#
# IMPORTANT:
# - No autoencoder retraining.
# - No basin centering.
# - No production files overwritten.
# - k=3 uses the FINAL production assignments.
# - k=2 is fitted only as a diagnostic comparison in the same standardized
#   raw 16-D latent space.

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import torch
except Exception:
    torch = None


# =============================================================================
# CONFIG
# =============================================================================

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
FINAL_CLUSTERS_FILE = "data/processed/window_clusters.parquet"
WAVELET_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"
MODEL_FILE = "models/temporal_autoencoder_v3.pt"

OUT_DIR = "results/tables/temporal_regime_fingerprint_audit"
FIG_DIR = "results/figures/temporal_regime_fingerprint_audit"

ASSIGNMENTS_OUT = os.path.join(OUT_DIR, "raw_k2_k3_assignments.parquet")
CALENDAR_OUT = os.path.join(OUT_DIR, "calendar_phase_dependence.csv")
TEMPLATE_COHERENCE_OUT = os.path.join(OUT_DIR, "temporal_template_coherence.csv")
TEMPLATE_PAIRWISE_OUT = os.path.join(OUT_DIR, "temporal_template_pairwise_similarity.csv")
DYNAMIC_PROFILES_OUT = os.path.join(OUT_DIR, "basin_balanced_dynamic_feature_profiles.csv")
DYNAMIC_SIGN_OUT = os.path.join(OUT_DIR, "dynamic_feature_sign_consistency.csv")
K2_METRICS_OUT = os.path.join(OUT_DIR, "raw_k2_diagnostic_metrics.csv")
K3_QC_OUT = os.path.join(OUT_DIR, "production_k3_reproduction_qc.csv")

WINDOW_LENGTH = 24
RANDOM_STATE = 42
N_INIT = 20
MIN_WINDOWS_PER_BASIN_STATE = 5

# The detailed temporal-template audit focuses on the core coupled
# climate-storage channels. They are all genuine autoencoder inputs.
PREFERRED_BASES = ["lwe_thickness", "tp", "swvl4"]
BANDS = ["short", "seasonal", "long"]

# Compact dynamic features printed to console.
FEATURES_TO_PRINT = [
    "lwe_thickness_long__delta",
    "lwe_thickness_long__slope",
    "lwe_thickness_long__range",
    "lwe_thickness_seasonal__rms",
    "lwe_thickness_short__rms",
    "tp_long__delta",
    "tp_short__rms",
    "swvl4_long__delta",
]

SIGN_FEATURES_TO_PRINT = [
    "lwe_thickness_long__delta",
    "lwe_thickness_long__slope",
    "tp_long__delta",
    "swvl4_long__delta",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def month_serial_from_ts(s):
    s = pd.to_datetime(s)
    return s.dt.year * 12 + s.dt.month


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a = a[ok]
    b = b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def row_center_scale(A):
    """
    Shape-normalize each row independently:
      subtract 24-month mean, divide by 24-month RMS/std.

    This removes absolute level and isolates temporal shape.
    """
    A = np.asarray(A, dtype=float)
    mu = np.nanmean(A, axis=1, keepdims=True)
    X = A - mu
    scale = np.sqrt(np.nanmean(X * X, axis=1, keepdims=True))
    scale[~np.isfinite(scale) | (scale < 1e-12)] = np.nan
    return X / scale


def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    v = v - np.nanmean(v)
    n = np.sqrt(np.nansum(v * v))
    if not np.isfinite(n) or n < 1e-12:
        return np.full_like(v, np.nan)
    return v / n


# =============================================================================
# LOAD INPUTS AND BUILD RAW k=2 / PRODUCTION k=3
# =============================================================================

def load_core():
    emb = pd.read_parquet(EMBEDDINGS_FILE).copy()
    clu = pd.read_parquet(FINAL_CLUSTERS_FILE).copy()

    zcols = latent_columns(emb)
    if len(zcols) != 16:
        raise ValueError(f"Expected 16 latent dimensions; found {len(zcols)}")

    required_emb = {"sample_id", "basin", "start_time", "end_time", "split"}
    miss = required_emb - set(emb.columns)
    if miss:
        raise ValueError(f"Continuous embeddings missing columns: {miss}")

    required_clu = {"sample_id", "cluster"}
    miss = required_clu - set(clu.columns)
    if miss:
        raise ValueError(f"Final cluster file missing columns: {miss}")

    if "k" in clu.columns:
        kvals = sorted(clu["k"].dropna().unique().tolist())
        if kvals != [3]:
            raise ValueError(f"Expected production k=3 only; found k={kvals}")

    emb["basin_id"] = emb["basin"].astype("int64")
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])

    d = emb.merge(
        clu[["sample_id", "cluster"]].rename(columns={"cluster": "raw_k3"}),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if d["raw_k3"].isna().any():
        raise ValueError("Some continuous embeddings lack production k=3 labels.")
    d["raw_k3"] = d["raw_k3"].astype(int)

    X = d[zcols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km2 = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    d["raw_k2"] = km2.fit_predict(Xs).astype(int)

    # Refit k=3 only as QC: production labels may be permuted but should match.
    km3_qc = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    k3_refit = km3_qc.fit_predict(Xs)
    ari = adjusted_rand_score(d["raw_k3"], k3_refit)

    # Geometry for k=2 only, on a fixed sample for speed.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(
        len(d),
        size=min(50000, len(d)),
        replace=False,
    )
    sil2 = silhouette_score(Xs[idx], d["raw_k2"].to_numpy()[idx])

    print(f"✅ Continuous windows: {len(d):,}")
    print(f"✅ Basins: {d['basin_id'].nunique():,}")
    print(f"✅ Production k=3 vs deterministic refit ARI: {ari:.6f}")
    print(f"✅ Diagnostic raw k=2 silhouette (50k sample): {sil2:.4f}")

    k2_metrics = pd.DataFrame([{
        "k": 2,
        "n_windows": len(d),
        "silhouette_sample_n": len(idx),
        "silhouette_score": sil2,
        "smallest_cluster_fraction": d["raw_k2"].value_counts(normalize=True).min(),
        "largest_cluster_fraction": d["raw_k2"].value_counts(normalize=True).max(),
    }])

    k3_qc = pd.DataFrame([{
        "production_vs_refit_ari": ari,
        "n_windows": len(d),
    }])

    return d, zcols, k2_metrics, k3_qc


# =============================================================================
# LOAD THE EXACT MODEL INPUT CHANNELS / SCALER
# =============================================================================

def detect_all_multiscale_channels(wavelet):
    return sorted(
        c for c in wavelet.columns
        if c.endswith("_short")
        or c.endswith("_seasonal")
        or c.endswith("_long")
    )


def load_checkpoint_metadata():
    if not os.path.exists(MODEL_FILE) or torch is None:
        return None, None

    try:
        # PyTorch >=2.6 defaults to weights_only=True; this checkpoint contains
        # metadata dictionaries, so request the complete trusted local object.
        try:
            ckpt = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(MODEL_FILE, map_location="cpu")

        channels = ckpt.get("channel_cols")
        scaler = ckpt.get("scaler")
        return channels, scaler
    except Exception as exc:
        warnings.warn(
            f"Could not read checkpoint metadata ({exc}). "
            "Will reproduce scaler from train-period rows."
        )
        return None, None


def reproduce_train_scaler(wavelet, core, channel_cols):
    """
    Fallback matching the training logic:
    train-only unique basin-month rows over the train-window period.
    """
    train = core[core["split"] == "train"].copy()
    if train.empty:
        raise ValueError("No train windows available for scaler fallback.")

    t0 = train["start_time"].min()
    t1 = train["end_time"].max()
    train_basins = set(train["basin_id"].unique())

    rows = wavelet[
        (wavelet["time"] >= t0)
        & (wavelet["time"] <= t1)
        & (wavelet["basin_id"].isin(train_basins))
    ]

    mean = rows[channel_cols].mean()
    std = rows[channel_cols].std(ddof=0).replace(0, np.nan)

    scaler = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "train_start": str(t0.date()),
        "train_end": str(t1.date()),
        "n_train_rows": int(len(rows)),
        "n_train_basins": int(len(train_basins)),
    }
    return scaler


def load_scaled_panel(core):
    w = pd.read_parquet(WAVELET_FILE).copy()
    if "basin" not in w.columns:
        raise ValueError("Wavelet dataset requires 'basin'.")

    if "time" in w.columns:
        w["time"] = pd.to_datetime(w["time"])
    elif {"year", "month"}.issubset(w.columns):
        w["time"] = pd.to_datetime(
            dict(year=w["year"], month=w["month"], day=1)
        )
    else:
        raise ValueError("Wavelet dataset requires time or year/month.")

    w["basin_id"] = w["basin"].astype("int64")
    w = w.drop_duplicates(["basin_id", "time"])
    w = w.sort_values(["basin_id", "time"]).reset_index(drop=True)

    all_detected = detect_all_multiscale_channels(w)
    checkpoint_channels, scaler = load_checkpoint_metadata()

    if checkpoint_channels is not None:
        channel_cols = list(checkpoint_channels)
        missing = [c for c in channel_cols if c not in w.columns]
        if missing:
            raise ValueError(
                f"Checkpoint input channels missing from wavelet file: {missing}"
            )
        print(
            f"✅ Loaded exact input channel list from checkpoint: "
            f"{len(channel_cols)} channels"
        )
    else:
        channel_cols = all_detected
        print(
            f"⚠️ Checkpoint channel metadata unavailable; using detected "
            f"{len(channel_cols)} multiscale channels"
        )

    if scaler is None:
        scaler = reproduce_train_scaler(w, core, channel_cols)
        print("⚠️ Reproduced train-only scaler because checkpoint scaler was unavailable.")
    else:
        print("✅ Loaded train-only channel scaler from checkpoint")

    for c in channel_cols:
        if c not in scaler["mean"] or c not in scaler["std"]:
            raise ValueError(f"Scaler missing channel {c}")
        sd = float(scaler["std"][c])
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Invalid scaler std for {c}: {sd}")
        w[c] = (
            w[c].astype(float) - float(scaler["mean"][c])
        ) / sd

    preferred = [
        f"{base}_{band}"
        for base in PREFERRED_BASES
        for band in BANDS
        if f"{base}_{band}" in channel_cols
    ]

    if "lwe_thickness_long" not in preferred:
        raise ValueError(
            "Core TWS long channel 'lwe_thickness_long' was not found among "
            "the autoencoder inputs."
        )

    if len(preferred) < 6:
        warnings.warn(
            f"Only {len(preferred)} preferred TWS/precip/deep-soil channels "
            f"were found: {preferred}"
        )

    print("✅ Detailed temporal-template channels:")
    for c in preferred:
        print(f"   {c}")

    w["_mserial"] = w["time"].dt.year * 12 + w["time"].dt.month
    panel = w.set_index(["basin_id", "_mserial"])[preferred].sort_index()

    return panel, preferred


# =============================================================================
# EXACT 24-MONTH SEQUENCE EXTRACTION
# =============================================================================

def extract_sequences(core, panel, key_channels):
    """
    Extract N x 24 x C arrays for the exact continuous windows.
    Memory is ~200 MB for 236k windows and 9 float32 channels.
    """
    basins = core["basin_id"].to_numpy(dtype=np.int64)
    end_serial = month_serial_from_ts(core["end_time"]).to_numpy(dtype=np.int64)
    n = len(core)
    c = len(key_channels)

    seq = np.empty((n, WINDOW_LENGTH, c), dtype=np.float32)

    print(
        f"🚀 Extracting exact 24-month model-input sequences "
        f"({n:,} windows × {c} key channels)..."
    )

    for pos, lag in enumerate(range(WINDOW_LENGTH - 1, -1, -1)):
        idx = pd.MultiIndex.from_arrays(
            [basins, end_serial - lag],
            names=["basin_id", "_mserial"],
        )
        vals = panel.reindex(idx)[key_channels].to_numpy(dtype=np.float32)
        seq[:, pos, :] = vals

        if pos in {0, 5, 11, 17, 23}:
            print(f"   relative month {pos + 1:02d}/24")

    finite = np.isfinite(seq).all(axis=(1, 2))
    bad = int((~finite).sum())

    if bad:
        examples = core.loc[
            ~finite,
            ["sample_id", "basin_id", "start_time", "end_time"],
        ].head(10)
        raise ValueError(
            f"{bad:,} continuous windows could not be reconstructed from the "
            f"scaled wavelet panel.\nExamples:\n{examples.to_string(index=False)}"
        )

    print("✅ All continuous windows reconstructed exactly with finite inputs.")
    return seq


# =============================================================================
# CALENDAR-PHASE DIAGNOSTIC
# =============================================================================

def calendar_dependence(core):
    rows = []

    start_month = core["start_time"].dt.month.to_numpy()
    end_month = core["end_time"].dt.month.to_numpy()

    for solution in ["raw_k2", "raw_k3"]:
        y = core[solution].to_numpy()

        for which, month in [
            ("start_month", start_month),
            ("end_month", end_month),
        ]:
            nmi = normalized_mutual_info_score(month, y)
            rows.append({
                "solution": solution,
                "calendar_variable": which,
                "normalized_mutual_information": float(nmi),
            })

    return pd.DataFrame(rows)


# =============================================================================
# TEMPORAL TEMPLATES
# =============================================================================

def basin_state_mean_sequences(core, seq_channel, solution):
    """
    For one input channel:
      window sequence -> average within basin×state.

    Requiring >=5 windows reduces fragile basin-state fingerprints.
    """
    cols = [f"m{i:02d}" for i in range(1, WINDOW_LENGTH + 1)]
    d = pd.DataFrame(seq_channel, columns=cols)
    d["basin_id"] = core["basin_id"].to_numpy()
    d["state"] = core[solution].to_numpy()

    g = d.groupby(["basin_id", "state"], observed=True)
    means = g[cols].mean()
    counts = g.size().rename("n_windows")
    means = means.join(counts).reset_index()
    means = means[
        means["n_windows"] >= MIN_WINDOWS_PER_BASIN_STATE
    ].reset_index(drop=True)

    return means, cols


def template_coherence_for_channel(core, seq_channel, solution, channel):
    """
    Shape-only temporal coherence.

    Each basin×state mean curve is centered/scaled over its 24 relative months.
    Global state templates are then basin-balanced means of those normalized
    curves.

    Metrics:
      - median own-template correlation
      - own-template wins fraction
      - fraction own correlation >0 / >0.5
    """
    bs, cols = basin_state_mean_sequences(core, seq_channel, solution)
    states = sorted(core[solution].unique().tolist())

    A = bs[cols].to_numpy(dtype=float)
    A_norm = row_center_scale(A)

    valid = np.isfinite(A_norm).all(axis=1)
    bs = bs.loc[valid].reset_index(drop=True)
    A_norm = A_norm[valid]

    templates = {}
    for state in states:
        mask = bs["state"].to_numpy() == state
        if mask.sum() == 0:
            templates[state] = np.full(WINDOW_LENGTH, np.nan)
            continue
        t = np.nanmean(A_norm[mask], axis=0)
        templates[state] = normalize_vector(t)

    corr_matrix = np.full((len(bs), len(states)), np.nan, dtype=float)
    for j, state in enumerate(states):
        t = templates[state]
        for i in range(len(bs)):
            corr_matrix[i, j] = safe_corr(A_norm[i], t)

    state_to_col = {state: j for j, state in enumerate(states)}
    own = np.asarray([
        corr_matrix[i, state_to_col[int(st)]]
        for i, st in enumerate(bs["state"].to_numpy())
    ])

    winner_idx = np.nanargmax(
        np.where(np.isfinite(corr_matrix), corr_matrix, -np.inf),
        axis=1,
    )
    winner_state = np.asarray([states[j] for j in winner_idx])
    own_win = winner_state == bs["state"].to_numpy()

    rows = []
    for state in states:
        mask = bs["state"].to_numpy() == state
        vals = own[mask]

        rows.append({
            "solution": solution,
            "channel": channel,
            "state": int(state),
            "n_basin_state_fingerprints": int(mask.sum()),
            "median_own_template_correlation": float(np.nanmedian(vals)),
            "mean_own_template_correlation": float(np.nanmean(vals)),
            "fraction_own_corr_positive": float(np.nanmean(vals > 0)),
            "fraction_own_corr_ge_0p5": float(np.nanmean(vals >= 0.5)),
            "own_template_win_fraction": float(np.nanmean(own_win[mask])),
        })

    pair_rows = []
    for i, a in enumerate(states):
        for b in states[i + 1:]:
            pair_rows.append({
                "solution": solution,
                "channel": channel,
                "state_a": int(a),
                "state_b": int(b),
                "template_correlation": safe_corr(
                    templates[a],
                    templates[b],
                ),
            })

    # Save channel template values for plotting/table inspection.
    template_rows = []
    for state in states:
        for month_idx, val in enumerate(templates[state], start=1):
            template_rows.append({
                "solution": solution,
                "channel": channel,
                "state": int(state),
                "relative_month": month_idx,
                "shape_template": val,
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(template_rows),
        bs,
        A_norm,
    )


def multichannel_template_coherence(core, channel_results, solution):
    """
    Concatenate the basin×state normalized 24-month curves for all key channels.
    Only basin×state combinations available for every key channel are used.
    """
    channel_maps = []

    for channel, item in channel_results.items():
        bs, A_norm = item["bs"], item["A_norm"]
        keys = list(zip(bs["basin_id"].astype(int), bs["state"].astype(int)))
        m = {key: A_norm[i] for i, key in enumerate(keys)}
        channel_maps.append((channel, m))

    common = set(channel_maps[0][1].keys())
    for _, m in channel_maps[1:]:
        common &= set(m.keys())

    common = sorted(common)
    if not common:
        raise ValueError(f"No common basin-state fingerprints for {solution}")

    features = []
    basins = []
    states_arr = []

    for basin_id, state in common:
        parts = []
        for _, m in channel_maps:
            parts.append(m[(basin_id, state)])
        features.append(np.concatenate(parts))
        basins.append(basin_id)
        states_arr.append(state)

    X = np.asarray(features, dtype=float)
    states_arr = np.asarray(states_arr, dtype=int)
    states = sorted(np.unique(states_arr).tolist())

    # Normalize flattened multichannel vectors to unit norm.
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    valid = np.isfinite(X).all(axis=1) & (norm[:, 0] > 1e-12)
    X = X[valid] / norm[valid]
    states_arr = states_arr[valid]
    basins = np.asarray(basins)[valid]

    templates = {}
    for state in states:
        t = X[states_arr == state].mean(axis=0)
        nt = np.linalg.norm(t)
        templates[state] = t / nt if nt > 1e-12 else np.full_like(t, np.nan)

    sim = np.column_stack([
        X @ templates[state]
        for state in states
    ])
    state_to_col = {state: j for j, state in enumerate(states)}
    own = np.asarray([
        sim[i, state_to_col[int(st)]]
        for i, st in enumerate(states_arr)
    ])
    winners = np.asarray([states[j] for j in np.argmax(sim, axis=1)])
    own_win = winners == states_arr

    rows = []
    for state in states:
        mask = states_arr == state
        vals = own[mask]
        rows.append({
            "solution": solution,
            "channel": "__MULTICHANNEL__",
            "state": int(state),
            "n_basin_state_fingerprints": int(mask.sum()),
            "median_own_template_correlation": float(np.median(vals)),
            "mean_own_template_correlation": float(np.mean(vals)),
            "fraction_own_corr_positive": float(np.mean(vals > 0)),
            "fraction_own_corr_ge_0p5": float(np.mean(vals >= 0.5)),
            "own_template_win_fraction": float(np.mean(own_win[mask])),
        })

    pair_rows = []
    for i, a in enumerate(states):
        for b in states[i + 1:]:
            pair_rows.append({
                "solution": solution,
                "channel": "__MULTICHANNEL__",
                "state_a": int(a),
                "state_b": int(b),
                "template_correlation": float(
                    np.dot(templates[a], templates[b])
                ),
            })

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


# =============================================================================
# DYNAMIC SUMMARY FEATURES
# =============================================================================

def sequence_feature_arrays(seq, key_channels):
    """
    Returns a dictionary {feature_name: N-vector}.

    All values are calculated from the exact standardized model-input sequence.
    """
    n, t, c = seq.shape
    x = np.arange(t, dtype=np.float32)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))

    out = {}

    for j, channel in enumerate(key_channels):
        A = seq[:, :, j].astype(np.float64)

        mean = A.mean(axis=1)
        start = A[:, 0]
        end = A[:, -1]
        delta = end - start
        slope = ((A - mean[:, None]) * xc[None, :]).sum(axis=1) / denom
        rng = A.max(axis=1) - A.min(axis=1)
        rms = np.sqrt(np.mean(A * A, axis=1))
        roughness = np.mean(np.abs(np.diff(A, axis=1)), axis=1)

        first8 = A[:, :8].mean(axis=1)
        mid8 = A[:, 8:16].mean(axis=1)
        last8 = A[:, 16:24].mean(axis=1)
        curvature = mid8 - 0.5 * (first8 + last8)
        late_minus_early = A[:, -6:].mean(axis=1) - A[:, :6].mean(axis=1)

        out[f"{channel}__mean"] = mean
        out[f"{channel}__delta"] = delta
        out[f"{channel}__slope"] = slope
        out[f"{channel}__range"] = rng
        out[f"{channel}__rms"] = rms
        out[f"{channel}__roughness"] = roughness
        out[f"{channel}__curvature"] = curvature
        out[f"{channel}__late6_minus_early6"] = late_minus_early

    return out


def basin_balanced_dynamic_profiles(core, feature_arrays, solution):
    """
    Window feature -> mean within basin×state -> global basin-balanced profile.
    """
    base = pd.DataFrame({
        "basin_id": core["basin_id"].to_numpy(),
        "state": core[solution].to_numpy(),
    })
    for name, values in feature_arrays.items():
        base[name] = values

    g = base.groupby(["basin_id", "state"], observed=True)
    br = g.mean().reset_index()
    counts = g.size().rename("n_windows").reset_index()
    br = br.merge(counts, on=["basin_id", "state"], how="left")
    br = br[
        br["n_windows"] >= MIN_WINDOWS_PER_BASIN_STATE
    ].reset_index(drop=True)

    rows = []
    feature_names = list(feature_arrays.keys())

    for state, s in br.groupby("state"):
        row = {
            "solution": solution,
            "state": int(state),
            "n_basins": int(s["basin_id"].nunique()),
        }
        for f in feature_names:
            row[f] = float(s[f].mean())
        rows.append(row)

    profiles = pd.DataFrame(rows).sort_values("state")

    sign_rows = []
    for state, s in br.groupby("state"):
        for f in feature_names:
            vals = s[f].to_numpy(dtype=float)
            pos = float(np.mean(vals > 0))
            neg = float(np.mean(vals < 0))
            zero = float(np.mean(np.isclose(vals, 0)))
            global_mean = float(np.mean(vals))

            sign_rows.append({
                "solution": solution,
                "state": int(state),
                "feature": f,
                "n_basins": len(vals),
                "basin_balanced_mean": global_mean,
                "basin_median": float(np.median(vals)),
                "fraction_positive": pos,
                "fraction_negative": neg,
                "fraction_near_zero": zero,
                "dominant_sign": (
                    "positive" if pos > neg
                    else "negative" if neg > pos
                    else "tie"
                ),
                "dominant_sign_fraction": max(pos, neg),
            })

    return profiles, pd.DataFrame(sign_rows)


# =============================================================================
# FIGURES
# =============================================================================

def save_template_plots(template_df, key_channels):
    for solution in ["raw_k2", "raw_k3"]:
        for channel in key_channels:
            sub = template_df[
                (template_df["solution"] == solution)
                & (template_df["channel"] == channel)
            ]
            if sub.empty:
                continue

            plt.figure(figsize=(9, 5))
            for state, s in sub.groupby("state"):
                plt.plot(
                    s["relative_month"],
                    s["shape_template"],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.5,
                    label=f"State {state}",
                )

            plt.axhline(0, linewidth=0.8)
            plt.xlabel("Relative month within 24-month window")
            plt.ylabel("Shape-normalized basin-balanced template")
            plt.title(f"{solution}: temporal fingerprint — {channel}")
            plt.legend()
            plt.tight_layout()

            out = os.path.join(
                FIG_DIR,
                f"{solution}_{channel}_temporal_template.png",
            )
            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

    print(f"✅ Saved temporal-template figures to: {FIG_DIR}")


def plot_multichannel_summary(coherence):
    sub = coherence[
        coherence["channel"] == "__MULTICHANNEL__"
    ].copy()
    if sub.empty:
        return

    sub["label"] = (
        sub["solution"]
        + " / state "
        + sub["state"].astype(str)
    )

    plt.figure(figsize=(9, 5))
    x = np.arange(len(sub))
    plt.bar(x, sub["own_template_win_fraction"])
    plt.xticks(x, sub["label"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Own-template win fraction")
    plt.title("Cross-basin coherence of multichannel 24-month dynamic shapes")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "multichannel_own_template_win_fraction.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def compact_feature(name):
    replacements = {
        "lwe_thickness_long": "TWS_long",
        "lwe_thickness_seasonal": "TWS_seasonal",
        "lwe_thickness_short": "TWS_short",
        "tp_long": "P_long",
        "tp_short": "P_short",
        "swvl4_long": "deep_soil_long",
    }
    for a, b in replacements.items():
        name = name.replace(a, b)
    return name


def print_summary(core, calendar, coherence, pairwise, profiles, sign):
    print("\n" + "=" * 94)
    print("TEMPORAL-REGIME FINGERPRINT AUDIT — RAW LATENT SPACE")
    print("=" * 94)

    print("\nA) State counts")
    for solution in ["raw_k2", "raw_k3"]:
        tab = (
            core[solution]
            .value_counts()
            .sort_index()
            .rename("n_windows")
            .to_frame()
        )
        tab["fraction"] = tab["n_windows"] / len(core)
        print(f"\n{solution}")
        print(tab.to_string(float_format=lambda x: f"{x: .3f}"))

    print("\nB) Calendar-phase dependence")
    print(
        calendar.to_string(
            index=False,
            float_format=lambda x: f"{x: .4f}",
        )
    )
    print(
        "   Interpretation: NMI near 0 means the states are not simply "
        "calendar-month labels."
    )

    print("\nC) MULTICHANNEL 24-month dynamic-template coherence")
    mc = coherence[coherence["channel"] == "__MULTICHANNEL__"].copy()
    print(
        mc[
            [
                "solution",
                "state",
                "n_basin_state_fingerprints",
                "median_own_template_correlation",
                "fraction_own_corr_ge_0p5",
                "own_template_win_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )

    print("\nD) MULTICHANNEL template separation")
    mp = pairwise[pairwise["channel"] == "__MULTICHANNEL__"].copy()
    print(
        mp.to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )
    print(
        "   Lower / negative pairwise template similarity means the recurrent "
        "states have more distinct temporal shapes."
    )

    print("\nE) Key channel-specific temporal-template coherence")
    key = coherence[
        coherence["channel"].isin([
            "lwe_thickness_short",
            "lwe_thickness_seasonal",
            "lwe_thickness_long",
            "tp_short",
            "tp_long",
            "swvl4_long",
        ])
    ].copy()

    print(
        key[
            [
                "solution", "channel", "state",
                "n_basin_state_fingerprints",
                "median_own_template_correlation",
                "own_template_win_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )

    print("\nF) Basin-balanced dynamic feature profiles")
    available = [f for f in FEATURES_TO_PRINT if f in profiles.columns]
    show = profiles[["solution", "state", "n_basins"] + available].copy()
    show = show.rename(columns={f: compact_feature(f) for f in available})
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )

    print("\nG) Dynamic-direction consistency across basins")
    s = sign[sign["feature"].isin(SIGN_FEATURES_TO_PRINT)].copy()
    if not s.empty:
        s["feature"] = s["feature"].map(compact_feature)
        print(
            s[
                [
                    "solution", "state", "feature", "n_basins",
                    "basin_balanced_mean",
                    "fraction_positive",
                    "fraction_negative",
                    "dominant_sign",
                    "dominant_sign_fraction",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x: .3f}",
            )
        )

    print("\nWhat would support 'recurrent dynamic regimes'?")
    print("- Calendar NMI should be low: states should not merely reproduce month-of-year.")
    print("- Multichannel own-template win fractions should be clearly above chance")
    print("  (chance = 0.50 for k=2; 0.33 for k=3).")
    print("- Own-template correlations should be meaningfully positive across basins.")
    print("- State templates should be distinguishable from one another.")
    print("- TWS/storage dynamic descriptors should reveal interpretable contrasts such as")
    print("  decline, recovery, persistence, variability, or seasonal response.")
    print("")
    print("STOP after this audit. Do not rename the states or retrain the autoencoder")
    print("until these temporal fingerprints are interpreted.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Temporal-regime fingerprint audit ---")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    core, zcols, k2_metrics, k3_qc = load_core()

    core[
        [
            "sample_id", "basin_id", "split",
            "start_time", "end_time", "raw_k2", "raw_k3",
        ]
    ].to_parquet(ASSIGNMENTS_OUT, index=False)
    k2_metrics.to_csv(K2_METRICS_OUT, index=False)
    k3_qc.to_csv(K3_QC_OUT, index=False)

    calendar = calendar_dependence(core)
    calendar.to_csv(CALENDAR_OUT, index=False)

    panel, key_channels = load_scaled_panel(core)
    seq = extract_sequences(core, panel, key_channels)

    # Dynamic summary features are calculated once from the same exact inputs.
    feature_arrays = sequence_feature_arrays(seq, key_channels)

    coherence_frames = []
    pair_frames = []
    template_frames = []
    profile_frames = []
    sign_frames = []

    for solution in ["raw_k2", "raw_k3"]:
        print(f"\n🚀 Temporal fingerprints: {solution}")

        channel_results = {}

        for j, channel in enumerate(key_channels):
            print(f"   channel: {channel}")

            coh, pair, templ, bs, A_norm = template_coherence_for_channel(
                core,
                seq[:, :, j],
                solution,
                channel,
            )

            coherence_frames.append(coh)
            pair_frames.append(pair)
            template_frames.append(templ)

            channel_results[channel] = {
                "bs": bs,
                "A_norm": A_norm,
            }

        mc_coh, mc_pair = multichannel_template_coherence(
            core,
            channel_results,
            solution,
        )
        coherence_frames.append(mc_coh)
        pair_frames.append(mc_pair)

        profiles, sign = basin_balanced_dynamic_profiles(
            core,
            feature_arrays,
            solution,
        )
        profile_frames.append(profiles)
        sign_frames.append(sign)

    coherence = pd.concat(coherence_frames, ignore_index=True)
    pairwise = pd.concat(pair_frames, ignore_index=True)
    templates = pd.concat(template_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    sign = pd.concat(sign_frames, ignore_index=True)

    coherence.to_csv(TEMPLATE_COHERENCE_OUT, index=False)
    pairwise.to_csv(TEMPLATE_PAIRWISE_OUT, index=False)
    profiles.to_csv(DYNAMIC_PROFILES_OUT, index=False)
    sign.to_csv(DYNAMIC_SIGN_OUT, index=False)

    templates_out = os.path.join(
        OUT_DIR,
        "basin_balanced_shape_templates.csv",
    )
    templates.to_csv(templates_out, index=False)

    save_template_plots(templates, key_channels)
    plot_multichannel_summary(coherence)

    print(f"\n✅ Saved: {ASSIGNMENTS_OUT}")
    print(f"✅ Saved: {CALENDAR_OUT}")
    print(f"✅ Saved: {TEMPLATE_COHERENCE_OUT}")
    print(f"✅ Saved: {TEMPLATE_PAIRWISE_OUT}")
    print(f"✅ Saved: {DYNAMIC_PROFILES_OUT}")
    print(f"✅ Saved: {DYNAMIC_SIGN_OUT}")
    print(f"✅ Saved: {templates_out}")

    print_summary(
        core,
        calendar,
        coherence,
        pairwise,
        profiles,
        sign,
    )

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
