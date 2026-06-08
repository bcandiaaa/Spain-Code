# -*- coding: utf-8 -*-
"""
TRNN for OFDM Signal Modulation Classification Using H5 Dataset Files

Spyder/Windows version.

This code:
- Loads 0dB.h5, 5dB.h5, 10dB.h5, 15dB.h5, 20dB.h5
- Combines all SNR datasets
- Trains a TRNN classifier
- Saves the model as trnn_ofdm_best.pt
- Loads trnn_ofdm_best.pt if it already exists
- Continues training every time you run it
- Does not stop epochs early
"""

import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =============================================================================
# 0. SPYDER PROJECT SETTINGS
# =============================================================================

PROJECT_DIR = r"C:\Users\usuario\TRNN_Project"

H5_FILE_NAMES = [
    "0dB.h5",
    "5dB.h5",
    "10dB.h5",
    "15dB.h5",
    "20dB.h5",
]

SAVE_NAME = "trnn_ofdm_best.pt"


# =============================================================================
# 1. CONSTANTS
# =============================================================================

MODULATIONS = [
    "BPSK+BPSK", "BPSK+QPSK", "BPSK+8PSK",
    "QPSK+BPSK", "QPSK+QPSK", "QPSK+8PSK",
]

NUM_CLASSES = len(MODULATIONS)
SAMPLING_LENGTH = 1024


# =============================================================================
# 2. TRNN NETWORK
# =============================================================================

def _conv_bn_relu(in_ch, out_ch, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualUnit(nn.Module):
    def __init__(self, channels, num_conv):
        super().__init__()

        layers = []

        for _ in range(num_conv):
            layers.append(
                _conv_bn_relu(
                    channels,
                    channels,
                    kernel=(3, 1),
                    padding=(1, 0)
                )
            )

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class TripleSkipResidualStack(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size):
        super().__init__()

        self.entry = _conv_bn_relu(
            in_ch,
            out_ch,
            kernel=(1, 1),
            padding=(0, 0)
        )

        self.ru_a = ResidualUnit(out_ch, num_conv=1)
        self.ru_b = ResidualUnit(out_ch, num_conv=2)
        self.ru_c = ResidualUnit(out_ch, num_conv=3)

        if in_ch != out_ch:
            self.proj = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(1, 1),
                bias=False
            )
        else:
            self.proj = nn.Identity()

        self.pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=pool_size
        )

    def forward(self, x):
        skip = self.proj(x)

        out = self.entry(x)
        out = self.ru_a(out)
        out = self.ru_b(out)
        out = self.ru_c(out)

        if skip.shape[2:] != out.shape[2:]:
            skip = F.adaptive_avg_pool2d(skip, out.shape[2:])

        return self.pool(out + skip)


class TRNN(nn.Module):
    """
    Input:
        (batch, 2, 1024)

    Model changes it internally to:
        (batch, 1, 2, 1024)

    Output:
        (batch, 6)
    """

    _POOL_SIZES = [(2, 2)] + [(1, 2)] * 6

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        blocks = []
        in_ch = 1

        for pool_size in self._POOL_SIZES:
            blocks.append(
                TripleSkipResidualStack(
                    in_ch,
                    32,
                    pool_size
                )
            )
            in_ch = 32

        self.trs = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.trs(x)
        x = self.gap(x)
        x = self.classifier(x)

        return x


# =============================================================================
# 3. DATASET PATHS FOR SPYDER
# =============================================================================

def get_h5_file_paths():
    """
    Finds the H5 files inside PROJECT_DIR.

    Example:
        C:\\Users\\usuario\\TRNN_Project\\0dB.h5
    """

    project_path = Path(PROJECT_DIR)

    if not project_path.exists():
        raise FileNotFoundError(
            f"\nProject folder not found:\n{project_path}\n\n"
            "Create this folder and place your .h5 files inside it."
        )

    h5_paths = []

    print("\nChecking H5 files inside:")
    print(project_path)

    for file_name in H5_FILE_NAMES:
        file_path = project_path / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"\nMissing file:\n{file_path}\n\n"
                f"Make sure {file_name} is inside:\n{project_path}"
            )

        print("Found:", file_path)
        h5_paths.append(str(file_path))

    return h5_paths


# =============================================================================
# 4. H5 DATASET LOADING
# =============================================================================

def load_h5_datasets(h5_files):
    """
    Loads multiple .h5 files and combines them.

    Expected inside each .h5 file:
        X      shape: (N, 1024, 2) or (N, 2, 1024)
        Y      shape: (1, N), (N, 1), or (N,)
        snrDb  one SNR value

    TRNN expects:
        X shape: (N, 2, 1024)
    """

    X_all = []
    y_all = []
    snr_all = []

    for file_path in h5_files:
        print("\nLoading file:", file_path)

        with h5py.File(file_path, "r") as f:
            print("Keys:", list(f.keys()))

            X = f["X"][:]
            y = f["Y"][:].squeeze()
            snr_value = float(f["snrDb"][:].squeeze())

        print("Original X shape:", X.shape)
        print("Original y shape:", y.shape)
        print("SNR value:", snr_value)

        if X.ndim == 3 and X.shape[1] == 1024 and X.shape[2] == 2:
            X = np.transpose(X, (0, 2, 1))

        elif X.ndim == 3 and X.shape[1] == 2 and X.shape[2] == 1024:
            pass

        else:
            raise ValueError(
                f"Unexpected X shape: {X.shape}. "
                "Expected (N, 1024, 2) or (N, 2, 1024)."
            )

        y = np.array(y).reshape(-1)

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch in {file_path}: len(X)={len(X)}, len(y)={len(y)}"
            )

        X_all.append(X.astype(np.float32))
        y_all.append(y.astype(np.int64))

        snr_array = np.full(len(y), snr_value)
        snr_all.append(snr_array)

        print("Converted X shape:", X.shape)
        print("Converted y shape:", y.shape)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    snr_all = np.concatenate(snr_all, axis=0)

    # If labels are 1 to 6, convert them to 0 to 5 for PyTorch CrossEntropyLoss.
    unique_labels = np.unique(y_all)

    if unique_labels.min() == 1 and unique_labels.max() == NUM_CLASSES:
        print("\nLabels appear to be 1 to 6. Converting them to 0 to 5.")
        y_all = y_all - 1

    print("\nCombined dataset:")
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)
    print("snr shape:", snr_all.shape)
    print("Classes found:", np.unique(y_all))
    print("SNR values found:", np.unique(snr_all))

    if np.min(y_all) < 0 or np.max(y_all) >= NUM_CLASSES:
        raise ValueError(
            f"Labels must be between 0 and {NUM_CLASSES - 1}. "
            f"Found labels: {np.unique(y_all)}"
        )

    return X_all, y_all, snr_all


class OFDMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# =============================================================================
# 5. TRAIN / VALIDATION SPLIT
# =============================================================================

def train_val_split(X, y, snr, train_ratio=0.70, seed=42):
    """
    Simple stratified split by class.
    This helps each class appear in both train and validation sets.
    """

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)

        n_train = int(train_ratio * len(class_indices))

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    snr_train = snr[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]
    snr_val = snr[val_indices]

    return X_train, y_train, snr_train, X_val, y_val, snr_val


# =============================================================================
# 6. CHECKPOINT LOADING AND SAVING
# =============================================================================

def load_checkpoint_if_exists(model, save_path, device):
    previous_best_acc = 0.0
    previous_total_epochs = 0

    if os.path.exists(save_path):
        print("\nFound existing checkpoint:")
        print(save_path)
        print("Loading previous model so training can continue...")

        checkpoint = torch.load(
            save_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        previous_best_acc = checkpoint.get("best_val_acc", 0.0)
        previous_total_epochs = checkpoint.get("total_epochs_trained", 0)

        print(f"Previous best validation accuracy: {previous_best_acc:.2%}")
        print(f"Previous total epochs trained: {previous_total_epochs}")

    else:
        print("\nNo previous checkpoint found.")
        print("Starting with a new TRNN model.")

    return model, previous_best_acc, previous_total_epochs


def save_checkpoint(
    model,
    save_path,
    best_val_acc,
    total_epochs_trained,
    current_run_epochs
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "modulations": MODULATIONS,
            "num_classes": NUM_CLASSES,
            "best_val_acc": best_val_acc,
            "total_epochs_trained": total_epochs_trained,
            "current_run_epochs": current_run_epochs,
            "model_name": "TRNN",
            "window_size": SAMPLING_LENGTH,
        },
        save_path
    )

    print("\nUpdated model saved:")
    print(save_path)


# =============================================================================
# 7. TRAINING
# =============================================================================

def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)

            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()

            total += len(yb)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-4,
    device="cpu",
    previous_best_acc=0.0
):
    """
    Trains for all epochs.
    No early stopping.
    """

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = previous_best_acc

    best_state = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
    }

    best_epoch_this_run = 0
    epochs_completed = 0

    print(
        f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>8} | {'Status':>12}"
    )
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        vl_loss, vl_acc = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device
        )

        scheduler.step()

        epochs_completed = epoch

        if vl_acc > best_acc:
            best_acc = vl_acc

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            best_epoch_this_run = epoch
            status = "improved"
        else:
            status = ""

        print(
            f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.2%} | "
            f"{vl_loss:>8.4f} | {vl_acc:>8.2%} | "
            f"{status:>12}"
        )

    model.load_state_dict(best_state)

    print(f"\nTraining completed for all {epochs_completed} epochs.")
    print(f"Best validation accuracy after this run: {best_acc:.2%}")

    return model, best_acc, best_epoch_this_run, epochs_completed


# =============================================================================
# 8. EVALUATION
# =============================================================================

def evaluate(model, loader, device):
    counts = {
        modulation: [0, 0]
        for modulation in MODULATIONS
    }

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=int
    )

    model.eval()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            preds = logits.argmax(1).cpu()

            for pred, true in zip(preds.tolist(), yb.tolist()):
                counts[MODULATIONS[true]][1] += 1
                cm[true][pred] += 1

                if pred == true:
                    counts[MODULATIONS[true]][0] += 1

    print("\nPer-class accuracy:")
    print("-" * 38)

    for name, (correct, total) in counts.items():
        acc = correct / total
        bar = "#" * int(acc * 20)

        print(f"  {name:<14}: {acc:6.2%}  {bar}")

    print("\nConfusion matrix:")
    print("Rows = true class, columns = predicted class")

    width = 14

    print(
        " " * width +
        "".join(f"{m:>{width}}" for m in MODULATIONS)
    )

    print("-" * (width * (NUM_CLASSES + 1)))

    for i, row_name in enumerate(MODULATIONS):
        print(
            f"{row_name:<{width}}" +
            "".join(
                f"{cm[i, j]:>{width}}"
                for j in range(NUM_CLASSES)
            )
        )


# =============================================================================
# 9. PLOTS
# =============================================================================

def plot_real_iq_example(X, y, snr, save_path):
    example_index = 0

    I = X[example_index, 0, :]
    Q = X[example_index, 1, :]

    label = int(y[example_index])
    snr_value = snr[example_index]

    plt.figure(figsize=(12, 5))
    plt.plot(I, label="I Channel")
    plt.plot(Q, label="Q Channel")
    plt.title(
        f"Real I/Q Signal Example: {MODULATIONS[label]} at {snr_value} dB"
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print("Saved:", save_path)


def plot_real_constellation_example(X, y, snr, save_path):
    example_index = 0

    I = X[example_index, 0, :]
    Q = X[example_index, 1, :]

    label = int(y[example_index])
    snr_value = snr[example_index]

    plt.figure(figsize=(6, 6))
    plt.scatter(I, Q, s=6, alpha=0.5)
    plt.title(
        f"Real I/Q Constellation: {MODULATIONS[label]} at {snr_value} dB"
    )
    plt.xlabel("In-phase I")
    plt.ylabel("Quadrature Q")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print("Saved:", save_path)


def plot_real_power_spectrum_example(X, y, snr, save_path):
    example_index = 0

    I = X[example_index, 0, :]
    Q = X[example_index, 1, :]

    complex_signal = I + 1j * Q

    label = int(y[example_index])
    snr_value = snr[example_index]

    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(complex_signal))
    )

    psd_db = 10 * np.log10(
        np.fft.fftshift(
            np.abs(np.fft.fft(complex_signal)) ** 2
        ) + 1e-12
    )

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, psd_db)
    plt.title(
        f"Real Power Spectral Density: {MODULATIONS[label]} at {snr_value} dB"
    )
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Power dB")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print("Saved:", save_path)


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    SEED = 42

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4

    project_path = Path(PROJECT_DIR)

    os.chdir(project_path)

    save_path = str(project_path / SAVE_NAME)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  TRNN — Real H5 OFDM Modulation Classification")
    print("=" * 60)
    print("  Running in Spyder / Windows")
    print(f"  Project folder: {project_path}")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Learning rate : {LR}")
    print(f"  Save path     : {save_path}")
    print("=" * 60)

    h5_files = get_h5_file_paths()

    X, y, snr = load_h5_datasets(h5_files)

    X_train, y_train, snr_train, X_val, y_val, snr_val = train_val_split(
        X,
        y,
        snr,
        train_ratio=0.70,
        seed=SEED
    )

    train_ds = OFDMDataset(X_train, y_train)
    val_ds = OFDMDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nDataset split:")
    print("Total samples:", len(y))
    print("Train samples:", len(train_ds))
    print("Val samples  :", len(val_ds))

    print("\nTraining SNR values:", np.unique(snr_train))
    print("Validation SNR values:", np.unique(snr_val))

    model = TRNN()

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTRNN parameters: {n_params:,}")

    model, previous_best_acc, previous_total_epochs = load_checkpoint_if_exists(
        model,
        save_path,
        device
    )

    model, best_val_acc, best_epoch_this_run, epochs_completed = train(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        previous_best_acc=previous_best_acc
    )

    total_epochs_trained = previous_total_epochs + epochs_completed

    evaluate(
        model,
        val_loader,
        device
    )

    save_checkpoint(
        model=model,
        save_path=save_path,
        best_val_acc=best_val_acc,
        total_epochs_trained=total_epochs_trained,
        current_run_epochs=epochs_completed
    )

    print(f"\nTotal epochs trained across runs: {total_epochs_trained}")

    print("\nGenerating plots...")

    plot_real_iq_example(
        X,
        y,
        snr,
        save_path=str(project_path / "real_iq_example.png")
    )

    plot_real_constellation_example(
        X,
        y,
        snr,
        save_path=str(project_path / "real_constellation_example.png")
    )

    plot_real_power_spectrum_example(
        X,
        y,
        snr,
        save_path=str(project_path / "real_power_spectrum_example.png")
    )

    print("\nDone.")


if __name__ == "__main__":
    main()