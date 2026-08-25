# trainTemporalAutoencoder.py
# Step 8: Temporal convolutional autoencoder for basin-window hydrological embeddings

import os
import json
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
DATA_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"

# Split-contained windows are used for fitting/evaluating the model.
WINDOW_METADATA_FILE = "data/processed/window_metadata_v3.parquet"

# Full-record windows are encoded after training and used for the scientific atlas.
CONTINUOUS_WINDOW_METADATA_FILE = "data/processed/window_metadata_continuous_v3.parquet"

MODEL_OUTPUT = "models/temporal_autoencoder_v3.pt"

# Evaluation embeddings retain train/val/test structure for ML diagnostics.
EMBEDDINGS_OUTPUT = "data/processed/embeddings_window_level_v3.parquet"

# Continuous embeddings are the canonical input for UMAP, regimes and trajectories.
CONTINUOUS_EMBEDDINGS_OUTPUT = "data/processed/embeddings_window_level_continuous_v3.parquet"

METRICS_OUTPUT = "results/tables/autoencoder_reconstruction_metrics_v3.csv"
CURVES_OUTPUT = "results/figures/training_curves_v3.png"
CONFIG_OUTPUT = "configs/temporal_autoencoder_v3.json"

WINDOW_LENGTH = 24
LATENT_DIM = 16
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATA
# =========================
def load_inputs():
    df = pd.read_parquet(DATA_FILE)
    eval_meta = pd.read_parquet(WINDOW_METADATA_FILE)
    continuous_meta = pd.read_parquet(CONTINUOUS_WINDOW_METADATA_FILE)

    print(f"✅ Loaded data: {DATA_FILE}")
    print(f"   Rows: {len(df):,}")
    print(f"✅ Loaded evaluation metadata: {WINDOW_METADATA_FILE}")
    print(f"   Windows: {len(eval_meta):,}")
    print(f"✅ Loaded continuous metadata: {CONTINUOUS_WINDOW_METADATA_FILE}")
    print(f"   Windows: {len(continuous_meta):,}")

    return df, eval_meta, continuous_meta


def detect_channel_columns(df):
    channels = sorted([
        c for c in df.columns
        if c.endswith("_short") or c.endswith("_seasonal") or c.endswith("_long")
    ])

    if not channels:
        raise ValueError("No multiscale channel columns found.")

    print(f"✅ Detected {len(channels)} channels")
    return channels


def prepare_panel(df, channel_cols):
    df = df.copy()
    df["time"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    keep_cols = ["basin", "year", "month", "time"] + channel_cols
    df = df[keep_cols]
    df = df.drop_duplicates(subset=["basin", "year", "month"])
    df = df.sort_values(["basin", "time"]).reset_index(drop=True)
    return df


def compute_train_scaler(panel_df, meta_df, channel_cols):
    """Fit the model-input scaler using only rows covered by train windows."""
    train_meta = meta_df[meta_df["split"] == "train"].copy()

    if train_meta.empty:
        raise ValueError("No train windows found in evaluation metadata.")

    train_start = pd.to_datetime(train_meta["start_time"]).min()
    train_end = pd.to_datetime(train_meta["end_time"]).max()

    train_rows = panel_df[
        (panel_df["time"] >= train_start) &
        (panel_df["time"] <= train_end)
    ].copy()

    train_basins = set(train_meta["basin"].unique())
    train_rows = train_rows[train_rows["basin"].isin(train_basins)]

    mean = train_rows[channel_cols].mean()
    std = train_rows[channel_cols].std(ddof=0).replace(0, np.nan)

    if std.isna().any():
        bad = list(std[std.isna()].index)
        raise ValueError(f"Zero/undefined training std for channels: {bad}")

    scaler = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "n_train_rows": int(len(train_rows)),
        "n_train_basins": int(len(train_basins)),
    }

    print("✅ Fitted train-only model-input scaler")
    print(f"   Train scaler period: {train_start.date()} to {train_end.date()}")
    print(f"   Train rows used: {len(train_rows):,}")
    print(f"   Train basins used: {len(train_basins):,}")
    return scaler


def apply_scaler(panel_df, channel_cols, scaler):
    panel_df = panel_df.copy()
    for c in channel_cols:
        panel_df[c] = (panel_df[c] - scaler["mean"][c]) / scaler["std"][c]
    return panel_df


class BasinWindowDataset(Dataset):
    def __init__(self, panel_df, metadata_df, channel_cols):
        self.metadata = metadata_df.reset_index(drop=True).copy()
        self.channel_cols = channel_cols
        self.lookup = {
            basin: g.sort_values("time").reset_index(drop=True)
            for basin, g in panel_df.groupby("basin")
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        basin = row["basin"]
        start_time = pd.to_datetime(row["start_time"])
        end_time = pd.to_datetime(row["end_time"])

        basin_df = self.lookup[basin]
        window = basin_df[
            (basin_df["time"] >= start_time) &
            (basin_df["time"] <= end_time)
        ].sort_values("time")

        x = window[self.channel_cols].to_numpy(dtype=np.float32)

        if x.shape != (WINDOW_LENGTH, len(self.channel_cols)):
            raise ValueError(f"Bad window shape for sample {row['sample_id']}: {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/inf found in sample {row['sample_id']}")

        return {"x": torch.from_numpy(x), "sample_id": int(row["sample_id"])}


# =========================
# MODEL
# =========================
class TemporalConvAutoencoder(nn.Module):
    def __init__(self, n_channels, latent_dim=16):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(16, latent_dim)

        self.from_latent = nn.Linear(latent_dim, 16 * WINDOW_LENGTH)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, n_channels, kernel_size=3, padding=1),
        )

    def encode(self, x):
        x = x.transpose(1, 2)  # batch, channels, time
        z = self.encoder_conv(x)
        z = self.pool(z).squeeze(-1)
        return self.to_latent(z)

    def decode(self, h):
        z = self.from_latent(h)
        z = z.view(h.size(0), 16, WINDOW_LENGTH)
        x_hat = self.decoder_conv(z)
        return x_hat.transpose(1, 2)

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h


# =========================
# TRAINING
# =========================
def make_loaders(panel_df, meta_df, channel_cols):
    loaders = {}

    for split in ["train", "val", "test"]:
        split_meta = meta_df[meta_df["split"] == split].copy()
        dataset = BasinWindowDataset(panel_df, split_meta, channel_cols)
        loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            drop_last=False,
        )
        print(f"✅ {split}: {len(dataset):,} evaluation windows")

    if len(loaders["train"].dataset) == 0 or len(loaders["val"].dataset) == 0:
        raise ValueError("Training and validation splits must both contain windows.")

    return loaders


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_values = 0
    criterion = nn.MSELoss(reduction="sum")

    for batch in loader:
        x = batch["x"].to(DEVICE)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_values += x.numel()

    return np.nan if n_values == 0 else total_loss / n_values


def train_model(model, loaders):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = []
    best_val = np.inf
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], optimizer)
        val_loss = run_epoch(model, loaders["val"], optimizer=None)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            # Deep copy is required so the stored best weights do not keep changing.
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("⏹️ Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def evaluate_splits(model, loaders):
    rows = []
    for split, loader in loaders.items():
        loss = run_epoch(model, loader, optimizer=None)
        rows.append({
            "split": split,
            "n_windows": len(loader.dataset),
            "mse": loss,
            "rmse": float(np.sqrt(loss)) if np.isfinite(loss) else np.nan,
        })

    metrics = pd.DataFrame(rows)
    print("\n✅ Reconstruction metrics:")
    print(metrics.to_string(index=False))
    return metrics


# =========================
# EXPORTS
# =========================
def export_embeddings(model, panel_df, metadata_df, channel_cols):
    dataset = BasinWindowDataset(panel_df, metadata_df, channel_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            sample_ids = batch["sample_id"].cpu().numpy()
            h = model.encode(x).cpu().numpy()

            for sid, vec in zip(sample_ids, h):
                row = {"sample_id": int(sid)}
                for j, val in enumerate(vec, start=1):
                    row[f"z{j}"] = float(val)
                all_rows.append(row)

    emb = pd.DataFrame(all_rows)
    metadata_cols = [c for c in metadata_df.columns if c != "sample_id"]
    emb = metadata_df[["sample_id"] + metadata_cols].merge(emb, on="sample_id", how="inner")
    return emb.sort_values("sample_id").reset_index(drop=True)


def save_embeddings(embeddings, output_file, label):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    embeddings.to_parquet(output_file, index=False)
    print(f"✅ Saved {label} embeddings: {output_file}")

    latent_cols = [c for c in embeddings.columns if c.startswith("z")]
    stds = embeddings[latent_cols].std()
    collapsed = int((stds < 1e-6).sum())
    print(f"   {label} latent collapse check: {collapsed} collapsed dimensions")
    return embeddings


def save_model(model, channel_cols, scaler):
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_channels": len(channel_cols),
        "latent_dim": LATENT_DIM,
        "window_length": WINDOW_LENGTH,
        "channel_cols": channel_cols,
        "scaler": scaler,
    }
    torch.save(checkpoint, MODEL_OUTPUT)
    print(f"✅ Saved model: {MODEL_OUTPUT}")


def save_metrics(metrics):
    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    metrics.to_csv(METRICS_OUTPUT, index=False)
    print(f"✅ Saved metrics: {METRICS_OUTPUT}")


def save_training_curves(history):
    os.makedirs(os.path.dirname(CURVES_OUTPUT), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Temporal Autoencoder Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved training curves: {CURVES_OUTPUT}")


def save_config(channel_cols):
    os.makedirs(os.path.dirname(CONFIG_OUTPUT), exist_ok=True)
    config = {
        "model": "TemporalConvAutoencoder",
        "input_shape": [WINDOW_LENGTH, len(channel_cols)],
        "latent_dim": LATENT_DIM,
        "channels": channel_cols,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "device": DEVICE,
        "input_file": DATA_FILE,
        "evaluation_window_metadata_file": WINDOW_METADATA_FILE,
        "continuous_window_metadata_file": CONTINUOUS_WINDOW_METADATA_FILE,
        "evaluation_embeddings_output": EMBEDDINGS_OUTPUT,
        "continuous_embeddings_output": CONTINUOUS_EMBEDDINGS_OUTPUT,
        "analysis_note": (
            "The autoencoder is fit only with split-contained evaluation windows. "
            "Continuous full-record windows are encoded after training for scientific "
            "state-space, clustering, trajectory, and transition analyses."
        ),
    }

    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved config: {CONFIG_OUTPUT}")


# =========================
# MAIN
# =========================
def main():
    print("--- Step 8: Temporal autoencoder representation learning ---")
    print(f"Device: {DEVICE}")
    set_seed(RANDOM_SEED)

    df, eval_meta, continuous_meta = load_inputs()
    channel_cols = detect_channel_columns(df)
    panel = prepare_panel(df, channel_cols)

    scaler = compute_train_scaler(panel, eval_meta, channel_cols)
    panel_scaled = apply_scaler(panel, channel_cols, scaler)

    # Train/evaluate only on split-contained windows.
    loaders = make_loaders(panel_scaled, eval_meta, channel_cols)
    model = TemporalConvAutoencoder(
        n_channels=len(channel_cols), latent_dim=LATENT_DIM
    ).to(DEVICE)
    model, history = train_model(model, loaders)
    metrics = evaluate_splits(model, loaders)

    save_model(model, channel_cols, scaler)
    save_metrics(metrics)
    save_training_curves(history)
    save_config(channel_cols)

    # Export the two deliberately distinct embedding products.
    eval_embeddings = export_embeddings(model, panel_scaled, eval_meta, channel_cols)
    save_embeddings(eval_embeddings, EMBEDDINGS_OUTPUT, "evaluation")

    continuous_embeddings = export_embeddings(
        model, panel_scaled, continuous_meta, channel_cols
    )
    save_embeddings(
        continuous_embeddings, CONTINUOUS_EMBEDDINGS_OUTPUT, "continuous-analysis"
    )

    print("\n✅ Separation of roles:")
    print("   evaluation embeddings -> reconstruction/generalization diagnostics")
    print("   continuous embeddings -> UMAP atlas, KMeans regimes, trajectories, transitions")
    print("--- Done ---")


if __name__ == "__main__":
    main()
