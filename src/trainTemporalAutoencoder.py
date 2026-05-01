# trainTemporalAutoencoder.py
# Step 8: Temporal convolutional autoencoder for basin-window hydrological embeddings

import os
import json
import random
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
WINDOW_METADATA_FILE = "data/processed/window_metadata_v3.parquet"

MODEL_OUTPUT = "models/temporal_autoencoder_v3.pt"
EMBEDDINGS_OUTPUT = "data/processed/embeddings_window_level_v3.parquet"
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
    meta = pd.read_parquet(WINDOW_METADATA_FILE)

    print(f"✅ Loaded data: {DATA_FILE}")
    print(f"   Rows: {len(df):,}")

    print(f"✅ Loaded metadata: {WINDOW_METADATA_FILE}")
    print(f"   Windows: {len(meta):,}")

    return df, meta


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
    """
    Fit channel scaler using unique basin-month rows covered by train windows.

    This is faster and cleaner than counting repeated overlapping-window rows.
    """
    train_meta = meta_df[meta_df["split"] == "train"].copy()

    if train_meta.empty:
        raise ValueError("No train windows found in metadata.")

    train_start = pd.to_datetime(train_meta["start_time"]).min()
    train_end = pd.to_datetime(train_meta["end_time"]).max()

    train_rows = panel_df[
        (panel_df["time"] >= train_start) &
        (panel_df["time"] <= train_end)
    ].copy()

    # Keep only basins that actually appear in train metadata
    train_basins = set(train_meta["basin"].unique())
    train_rows = train_rows[train_rows["basin"].isin(train_basins)]

    mean = train_rows[channel_cols].mean()
    std = train_rows[channel_cols].std(ddof=0).replace(0, np.nan)

    scaler = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "n_train_rows": int(len(train_rows)),
        "n_train_basins": int(len(train_basins)),
    }

    print("✅ Fitted train-only channel scaler from unique basin-month rows")
    print(f"   Train scaler period: {train_start.date()} to {train_end.date()}")
    print(f"   Train rows used: {len(train_rows):,}")
    print(f"   Train basins used: {len(train_basins):,}")

    return scaler

def apply_scaler(panel_df, channel_cols, scaler):
    panel_df = panel_df.copy()

    for c in channel_cols:
        mean = scaler["mean"][c]
        std = scaler["std"][c]
        panel_df[c] = (panel_df[c] - mean) / std

    return panel_df


class BasinWindowDataset(Dataset):
    def __init__(self, panel_df, metadata_df, channel_cols):
        self.panel_df = panel_df
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
            raise ValueError(
                f"Bad window shape for sample {row['sample_id']}: {x.shape}"
            )

        if not np.isfinite(x).all():
            raise ValueError(f"NaN/inf found in sample {row['sample_id']}")

        return {
            "x": torch.from_numpy(x),
            "sample_id": int(row["sample_id"]),
        }


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
        # x: batch, time, channels
        x = x.transpose(1, 2)  # batch, channels, time
        z = self.encoder_conv(x)
        z = self.pool(z).squeeze(-1)
        h = self.to_latent(z)
        return h

    def decode(self, h):
        z = self.from_latent(h)
        z = z.view(h.size(0), 16, WINDOW_LENGTH)
        x_hat = self.decoder_conv(z)
        x_hat = x_hat.transpose(1, 2)  # batch, time, channels
        return x_hat

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


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

        print(f"✅ {split}: {len(dataset):,} windows")

    return loaders


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples = 0

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
        n_samples += x.numel()

    return total_loss / n_samples


def train_model(model, loaders):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_val = np.inf
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], optimizer)
        val_loss = run_epoch(model, loaders["val"], optimizer)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
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
            "mse": loss,
            "rmse": float(np.sqrt(loss)),
        })

    metrics = pd.DataFrame(rows)
    print("\n✅ Reconstruction metrics:")
    print(metrics.to_string(index=False))

    return metrics


# =========================
# EXPORTS
# =========================
def export_embeddings(model, loader, metadata_df):
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

    meta_cols = ["sample_id", "basin", "start_time", "end_time", "split"]
    emb = metadata_df[meta_cols].merge(emb, on="sample_id", how="inner")

    emb = emb.sort_values("sample_id").reset_index(drop=True)
    return emb


def export_all_embeddings(model, panel_df, meta_df, channel_cols):
    full_dataset = BasinWindowDataset(panel_df, meta_df, channel_cols)

    full_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    embeddings = export_embeddings(model, full_loader, meta_df)

    os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT), exist_ok=True)
    embeddings.to_parquet(EMBEDDINGS_OUTPUT, index=False)

    print(f"✅ Saved embeddings: {EMBEDDINGS_OUTPUT}")

    latent_cols = [c for c in embeddings.columns if c.startswith("z")]
    stds = embeddings[latent_cols].std()

    print("\nEmbedding collapse check — latent std:")
    print(stds)

    collapsed = (stds < 1e-6).sum()
    if collapsed == 0:
        print("✅ No collapsed latent dimensions detected.")
    else:
        print(f"⚠️ {collapsed} latent dimensions appear collapsed.")

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
        "window_metadata_file": WINDOW_METADATA_FILE,
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

    df, meta = load_inputs()
    channel_cols = detect_channel_columns(df)

    panel = prepare_panel(df, channel_cols)

    scaler = compute_train_scaler(panel, meta, channel_cols)
    panel = apply_scaler(panel, channel_cols, scaler)

    loaders = make_loaders(panel, meta, channel_cols)

    model = TemporalConvAutoencoder(
        n_channels=len(channel_cols),
        latent_dim=LATENT_DIM,
    ).to(DEVICE)

    model, history = train_model(model, loaders)

    metrics = evaluate_splits(model, loaders)

    save_model(model, channel_cols, scaler)
    save_metrics(metrics)
    save_training_curves(history)
    save_config(channel_cols)

    export_all_embeddings(model, panel, meta, channel_cols)

    print("--- Done ---")


if __name__ == "__main__":
    main()