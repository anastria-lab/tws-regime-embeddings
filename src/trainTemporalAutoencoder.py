# trainTemporalAutoencoder.py
# Step 8: Temporal convolutional autoencoder for basin-window hydrological embeddings

import os
import json
import random
import copy
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# RTX/Turing stability setting.  The local RTX 2070 SUPER + PyTorch/CUDA stack
# showed low-level cuDNN failures during long runs even when single-batch smoke
# tests passed.  Disable cuDNN for this compact stride-1 Conv1d model while
# retaining CUDA execution for tensors, linear layers, activations and native
# convolution kernels.  This removes the need for CUDA_LAUNCH_BLOCKING=1.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =========================
# CONFIG
# =========================
# Canonical retrospective multiscale feature product used for BOTH model fitting
# diagnostics and the scientific atlas. This avoids training the encoder on a
# different SWT definition from the one it later embeds.
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
ORDER_SENSITIVITY_OUTPUT = "results/tables/temporal_order_sensitivity_v3.csv"

WINDOW_LENGTH = 24
LATENT_DIM = 16
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
RANDOM_SEED = 42

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE =  "cpu"


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


# =========================
# DATA
# =========================
def load_inputs():
    df = pd.read_parquet(DATA_FILE)
    eval_meta = pd.read_parquet(WINDOW_METADATA_FILE)
    continuous_meta = pd.read_parquet(CONTINUOUS_WINDOW_METADATA_FILE)

    print(f"✅ Loaded canonical retrospective SWT data: {DATA_FILE}")
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
    """Fast month-indexed window dataset.

    The previous implementation performed pandas ``iloc`` + boolean date filtering
    + ``to_numpy`` for every single training sample.  With ~173k train windows this
    made the data loader the dominant runtime.  Here each basin is converted once
    to one contiguous float32 tensor and every metadata row is resolved once to an
    integer start position.  ``__getitem__`` is then only a 24-row tensor slice.
    """
    def __init__(self, panel_df, metadata_df, channel_cols):
        self.metadata = metadata_df.reset_index(drop=True).copy()
        self.channel_cols = channel_cols
        self.sample_ids = self.metadata["sample_id"].to_numpy(dtype=np.int64)

        # Convert each basin panel to a contiguous CPU tensor only once.
        self.values_by_basin = []
        self.first_month_key = {}
        self.basin_to_idx = {}

        for basin, g in panel_df.groupby("basin", sort=False):
            g = g.sort_values("time")
            month_key = (g["time"].dt.year.to_numpy(dtype=np.int32) * 12
                         + g["time"].dt.month.to_numpy(dtype=np.int32))
            if len(month_key) > 1 and not np.all(np.diff(month_key) == 1):
                raise ValueError(f"Panel is not a complete monthly calendar for basin {basin}.")

            values = np.ascontiguousarray(
                g[channel_cols].to_numpy(dtype=np.float32, copy=True)
            )
            bidx = len(self.values_by_basin)
            self.basin_to_idx[basin] = bidx
            self.first_month_key[basin] = int(month_key[0])
            self.values_by_basin.append(torch.from_numpy(values))

        # Resolve every metadata row to (basin tensor index, integer start row)
        # once during Dataset construction.
        n = len(self.metadata)
        self.basin_indices = np.empty(n, dtype=np.int32)
        self.start_positions = np.empty(n, dtype=np.int32)

        starts = pd.to_datetime(self.metadata["start_time"])
        ends = pd.to_datetime(self.metadata["end_time"])
        start_keys = (starts.dt.year.to_numpy(dtype=np.int32) * 12
                      + starts.dt.month.to_numpy(dtype=np.int32))
        end_keys = (ends.dt.year.to_numpy(dtype=np.int32) * 12
                    + ends.dt.month.to_numpy(dtype=np.int32))

        if not np.all(end_keys - start_keys == WINDOW_LENGTH - 1):
            bad = int(np.sum((end_keys - start_keys) != WINDOW_LENGTH - 1))
            raise ValueError(f"Found {bad} metadata windows that are not exactly {WINDOW_LENGTH} months.")

        basin_values = self.metadata["basin"].to_numpy()
        for basin, row_idx in self.metadata.groupby("basin", sort=False).groups.items():
            if basin not in self.basin_to_idx:
                raise KeyError(f"Basin {basin} from metadata is absent from the panel.")
            idx = np.asarray(row_idx, dtype=np.int64)
            bidx = self.basin_to_idx[basin]
            pos = start_keys[idx] - self.first_month_key[basin]
            max_start = self.values_by_basin[bidx].shape[0] - WINDOW_LENGTH
            if np.any(pos < 0) or np.any(pos > max_start):
                raise ValueError(f"Window bounds fall outside the panel for basin {basin}.")
            self.basin_indices[idx] = bidx
            self.start_positions[idx] = pos.astype(np.int32)

        # One pass over the resolved windows catches NaN/inf or accidental gaps
        # before training starts.  This is much cheaper than pandas filtering every
        # window on every epoch.
        for basin, row_idx in self.metadata.groupby("basin", sort=False).groups.items():
            idx = np.asarray(row_idx, dtype=np.int64)
            bidx = self.basin_to_idx[basin]
            arr = self.values_by_basin[bidx]
            for p in self.start_positions[idx]:
                x = arr[int(p):int(p) + WINDOW_LENGTH]
                if x.shape != (WINDOW_LENGTH, len(self.channel_cols)):
                    raise ValueError(f"Bad cached window shape for basin {basin}: {tuple(x.shape)}")
                if not torch.isfinite(x).all():
                    raise ValueError(f"NaN/inf found in cached window for basin {basin}.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        bidx = int(self.basin_indices[idx])
        start = int(self.start_positions[idx])
        x = self.values_by_basin[bidx][start:start + WINDOW_LENGTH]
        return {"x": x, "sample_id": int(self.sample_ids[idx])}


# =========================
# MODEL
# =========================
class TemporalConvAutoencoder(nn.Module):
    """
    Temporal-order-preserving autoencoder using only stride-1 Conv1d kernels.

    On the target RTX 2070 SUPER, a basic stride-1 Conv1d forward/backward test
    succeeds whereas the previous downsampling/transposed-convolution CUDA paths
    can trigger cudaErrorMisalignedAddress. To avoid those backend paths entirely,
    this encoder keeps all 24 temporal positions and flattens the final 16 x 24
    feature map before the 16-D bottleneck.

    This preserves temporal order without global average pooling or strided
    convolutions. The decoder uses dense temporal expansion followed only by
    stride-1 Conv1d refinement.
    """
    def __init__(self, n_channels, latent_dim=16):
        super().__init__()

        # Deliberately restricted to the CUDA path already verified on the
        # target GPU: stride=1, kernel_size=3, padding=1 Conv1d.
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Flattening 16 x 24 retains position information. This is the key
        # difference from the original AdaptiveAvgPool1d(1) architecture.
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * WINDOW_LENGTH, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * WINDOW_LENGTH),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, n_channels, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x):
        x = x.transpose(1, 2)  # batch, channels, time
        z = self.encoder_conv(x)
        return self.to_latent(z)

    def decode(self, h):
        z = self.from_latent(h).view(h.size(0), 16, WINDOW_LENGTH)
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
    pin = DEVICE == "cuda"

    for split in ["train", "val", "test"]:
        split_meta = meta_df[meta_df["split"] == split].copy()
        t0 = time.perf_counter()
        dataset = BasinWindowDataset(panel_df, split_meta, channel_cols)
        loaders[split] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,          # robust on Windows; indexing is now very cheap
            pin_memory= False,
            drop_last=False,
        )
        print(
            f"✅ {split}: {len(dataset):,} evaluation windows | "
            f"cached/indexed in {time.perf_counter() - t0:.1f}s"
        )

    if len(loaders["train"].dataset) == 0 or len(loaders["val"].dataset) == 0:
        raise ValueError("Training and validation splits must both contain windows.")

    return loaders


def run_epoch(model, loader, optimizer=None, label="epoch"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    # Keep loss accumulation on-device and synchronize only for sparse progress
    # messages and once at the end.  The previous loss.item() on every batch forced
    # a CUDA synchronization hundreds of times per epoch.
    total_loss = torch.zeros((), device=DEVICE, dtype=torch.float64)
    n_values = 0
    criterion = nn.MSELoss(reduction="sum")
    pin = DEVICE == "cuda"
    t0 = time.perf_counter()
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(DEVICE, non_blocking=pin)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.detach().to(torch.float64)
        n_values += x.numel()

        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == n_batches:
            elapsed = time.perf_counter() - t0
            rate = batch_idx / elapsed if elapsed > 0 else float("nan")
            eta = (n_batches - batch_idx) / rate if rate > 0 else float("nan")
            print(
                f"   {label}: batch {batch_idx:,}/{n_batches:,} | "
                f"elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min"
            )

    value = total_loss.item()
    return np.nan if n_values == 0 else value / n_values


def train_model(model, loaders):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = []
    best_val = np.inf
    best_state = None
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, loaders["train"], optimizer, label=f"train e{epoch:03d}")
        val_loss = run_epoch(model, loaders["val"], optimizer=None, label=f"val e{epoch:03d}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            # Deep copy is required so the stored best weights do not keep changing.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        loss = run_epoch(model, loader, optimizer=None, label=f"eval {split}")
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


def evaluate_temporal_order_sensitivity(model, panel_df, metadata_df, channel_cols, sample_size=4096):
    """
    Diagnostic only: compare latent changes caused by reversing or cyclically
    shifting the same 24-month window with distances between unrelated windows.

    A ratio near zero would indicate near-invariance to temporal order. Values
    appreciably above zero show that temporal arrangement affects the embedding.
    No universal pass/fail threshold is imposed; save the values for comparison.
    """
    if len(metadata_df) == 0:
        return pd.DataFrame()

    n = min(sample_size, len(metadata_df))
    meta = metadata_df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    ds = BasinWindowDataset(panel_df, meta, channel_cols)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == "cuda"))

    d_reverse = []
    d_shift6 = []
    all_z = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=(DEVICE == "cuda"))
            z = model.encode(x)
            z_reverse = model.encode(torch.flip(x, dims=[1]))
            z_shift6 = model.encode(torch.roll(x, shifts=6, dims=1))

            d_reverse.extend(torch.linalg.vector_norm(z - z_reverse, dim=1).cpu().numpy())
            d_shift6.extend(torch.linalg.vector_norm(z - z_shift6, dim=1).cpu().numpy())
            all_z.append(z.cpu())

    z = torch.cat(all_z, dim=0)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = torch.as_tensor(rng.permutation(len(z)), dtype=torch.long)
    d_random = torch.linalg.vector_norm(z - z[perm], dim=1).numpy()

    reverse_mean = float(np.mean(d_reverse))
    shift_mean = float(np.mean(d_shift6))
    random_mean = float(np.mean(d_random))

    rows = [
        {"comparison": "reverse_time", "mean_latent_distance": reverse_mean,
         "relative_to_random_window_distance": reverse_mean / random_mean if random_mean else np.nan},
        {"comparison": "cyclic_shift_6_months", "mean_latent_distance": shift_mean,
         "relative_to_random_window_distance": shift_mean / random_mean if random_mean else np.nan},
        {"comparison": "random_other_window", "mean_latent_distance": random_mean,
         "relative_to_random_window_distance": 1.0},
    ]
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(ORDER_SENSITIVITY_OUTPUT), exist_ok=True)
    out.to_csv(ORDER_SENSITIVITY_OUTPUT, index=False)
    print("\n✅ Temporal-order sensitivity diagnostic:")
    print(out.to_string(index=False))
    print(f"✅ Saved: {ORDER_SENSITIVITY_OUTPUT}")
    return out


# =========================
# EXPORTS
# =========================
def export_embeddings(model, panel_df, metadata_df, channel_cols):
    t0 = time.perf_counter()
    dataset = BasinWindowDataset(panel_df, metadata_df, channel_cols)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    model.eval()
    all_ids = []
    all_z = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(DEVICE, non_blocking=(DEVICE == "cuda"))
            z = model.encode(x)
            all_ids.append(batch["sample_id"].numpy())
            all_z.append(z.cpu().numpy())
            if batch_idx % 200 == 0 or batch_idx == len(loader):
                print(f"   embedding export: batch {batch_idx:,}/{len(loader):,}")

    ids = np.concatenate(all_ids).astype(np.int64, copy=False)
    Z = np.concatenate(all_z, axis=0)
    latent_cols = [f"z{i}" for i in range(1, Z.shape[1] + 1)]
    latent_df = pd.DataFrame(Z, columns=latent_cols)
    latent_df.insert(0, "sample_id", ids)

    metadata_cols = [c for c in metadata_df.columns if c != "sample_id"]
    emb = metadata_df[["sample_id"] + metadata_cols].merge(
        latent_df, on="sample_id", how="inner", validate="one_to_one"
    )
    print(f"   embedding export completed in {(time.perf_counter() - t0)/60:.1f} min")
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
        "model": "TemporalConvAutoencoder_OrderPreserving_Stride1",
        "input_shape": [WINDOW_LENGTH, len(channel_cols)],
        "latent_dim": LATENT_DIM,
        "channels": channel_cols,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "device": DEVICE,
        "evaluation_input_file": DATA_FILE,
        "atlas_input_file": DATA_FILE,
        "evaluation_window_metadata_file": WINDOW_METADATA_FILE,
        "continuous_window_metadata_file": CONTINUOUS_WINDOW_METADATA_FILE,
        "evaluation_embeddings_output": EMBEDDINGS_OUTPUT,
        "continuous_embeddings_output": CONTINUOUS_EMBEDDINGS_OUTPUT,
        "architecture_note": (
            "Temporal order is preserved by retaining all 24 positions through "
            "stride-1 Conv1d layers and flattening the 16x24 temporal feature map "
            "before the 16-D bottleneck. No global average pooling, strided Conv1d, "
            "or ConvTranspose1d is used."
        ),
        "runtime_note": (
            "Fast cached month-indexed window loading; CUDA uses native kernels with cuDNN disabled "
            "for RTX 2070 SUPER stability. CUDA_LAUNCH_BLOCKING is not required."
        ),
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
# CUDA SMOKE TEST
# =========================
def run_cuda_smoke_test(n_channels=27, batch_size=256):
    """Exercise the exact model on full and partial batch sizes."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment.")

    set_seed(RANDOM_SEED)
    device = torch.device("cuda")
    model = TemporalConvAutoencoder(n_channels=n_channels, latent_dim=LATENT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for bs in [batch_size, 104, 64, 16, 1]:
        opt.zero_grad(set_to_none=True)
        x = torch.randn(bs, WINDOW_LENGTH, n_channels, device=device)
        x_hat, z = model(x)
        loss = torch.mean((x_hat - x) ** 2)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        print(
            f"✅ CUDA smoke batch={bs}: input={tuple(x.shape)} "
            f"latent={tuple(z.shape)} recon={tuple(x_hat.shape)} loss={loss.item():.6f}"
        )

    print("✅ EXACT MODEL MULTI-BATCH CUDA SMOKE TEST SUCCESS")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")


# =========================
# MAIN
# =========================
def main():
    print("--- Step 8: Fast/stable stride-1 temporal-order-preserving autoencoder ---")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled} (disabled intentionally for stability)")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        print("⚠️ CUDA_LAUNCH_BLOCKING=1 is set. Clear it for production training; this script disables cuDNN instead.")
    set_seed(RANDOM_SEED)

    df, eval_meta, continuous_meta = load_inputs()

    channel_cols = detect_channel_columns(df)
    panel = prepare_panel(df, channel_cols)

    # The model-input scaler is still fitted from train-window rows only.
    scaler = compute_train_scaler(panel, eval_meta, channel_cols)
    panel_scaled = apply_scaler(panel, channel_cols, scaler)

    # Split-contained windows provide reconstruction diagnostics. The feature
    # definition itself is the same retrospective SWT used by the atlas.
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

    # Verify that the learned representation responds to temporal arrangement.
    evaluate_temporal_order_sensitivity(
        model, panel_scaled, continuous_meta, channel_cols
    )

    eval_embeddings = export_embeddings(
        model, panel_scaled, eval_meta, channel_cols
    )
    save_embeddings(eval_embeddings, EMBEDDINGS_OUTPUT, "evaluation")

    continuous_embeddings = export_embeddings(
        model, panel_scaled, continuous_meta, channel_cols
    )
    save_embeddings(
        continuous_embeddings, CONTINUOUS_EMBEDDINGS_OUTPUT, "continuous-analysis"
    )

    print("\n✅ Atlas-consistent modeling:")
    print("   train/val/test reconstruction <- retrospective SWT, split-contained windows")
    print("   continuous scientific atlas <- the SAME retrospective SWT feature product")
    print("   climatology/std and model-input scaler <- train/reference period only")
    print("   reconstruction metrics are diagnostics, NOT forecasting/generalization claims")
    print("--- Done ---")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        run_cuda_smoke_test()
    else:
        main()
