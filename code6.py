"""
Improved TRNN Modulation Classifier
BPSK vs QPSK vs 8PSK

Includes:
- Early stopping
- Frame normalization
- Smaller model to reduce overfitting
- AdamW optimizer
- Gradient clipping
- Balanced train/validation/test split by SNR
- Accuracy by SNR
- Clean constellation plots
- Clear time-domain I/Q signal plots

Install:
    pip install torch matplotlib scikit-learn
"""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

class Config:
    SAMPLES_PER_CLASS: int = 6000
    SEQ_LEN: int = 512

    # Easier range for better validation accuracy.
    # After this works, you can change it to (-10, 22, 2).
    SNR_RANGE_DB: Tuple[int, int, int] = (0, 22, 2)

    NUM_CLASSES: int = 3
    BASE_FILTERS: int = 32
    DROPOUT: float = 0.25

    EPOCHS: int = 60
    BATCH_SIZE: int = 64
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.05

    EARLY_STOP_PATIENCE: int = 8
    SEED: int = 42

    OUT_DIR: Path = Path("outputs")
    CHECKPOINT: Path = Path("outputs/trnn_best_with_time_domain.pt")
    CACHE: Path = Path("outputs/dataset_with_time_domain.pt")

    FORCE_REGENERATE_DATA: bool = False

    CLASSES: List[str] = ["BPSK", "QPSK", "8PSK"]
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
cfg.OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# SIGNAL GENERATION
# ============================================================

def randn_pair():
    """
    Generates two Gaussian random values using the Box-Muller transform.
    Used for I and Q noise.
    """
    u1 = random.random() or 1e-12
    u2 = random.random()

    r = math.sqrt(-2.0 * math.log(u1))

    return r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)


def add_awgn(I: float, Q: float, snr_db: float):
    """
    Adds complex AWGN noise to an I/Q symbol.
    """
    noise_std = math.pow(10.0, -snr_db / 20.0) * math.sqrt(0.5)

    ni, nq = randn_pair()

    return I + ni * noise_std, Q + nq * noise_std


def normalize_frame(I_ch, Q_ch):
    """
    Normalizes each I/Q frame so the average power is close to 1.
    This helps training and reduces overfitting.
    """
    power = math.sqrt(
        sum(i * i + q * q for i, q in zip(I_ch, Q_ch)) / len(I_ch)
    )

    if power < 1e-12:
        power = 1e-12

    I_ch = [i / power for i in I_ch]
    Q_ch = [q / power for q in Q_ch]

    return I_ch, Q_ch


def gen_bpsk_frame(seq_len: int, snr_db: float, normalize: bool = True):
    """
    BPSK:
    Two points on the I-axis.
    Symbols: +1 and -1
    Q is ideally 0.
    """
    I_ch, Q_ch = [], []

    for _ in range(seq_len):
        symbol = 1.0 if random.random() < 0.5 else -1.0

        i, q = add_awgn(symbol, 0.0, snr_db)

        I_ch.append(i)
        Q_ch.append(q)

    if normalize:
        return normalize_frame(I_ch, Q_ch)

    return I_ch, Q_ch


def gen_qpsk_frame(seq_len: int, snr_db: float, normalize: bool = True):
    """
    QPSK:
    Four points in the I/Q plane.
    """
    inv = 1.0 / math.sqrt(2.0)

    symbols = [
        (inv, inv),
        (-inv, inv),
        (-inv, -inv),
        (inv, -inv),
    ]

    I_ch, Q_ch = [], []

    for _ in range(seq_len):
        si, sq = random.choice(symbols)

        i, q = add_awgn(si, sq, snr_db)

        I_ch.append(i)
        Q_ch.append(q)

    if normalize:
        return normalize_frame(I_ch, Q_ch)

    return I_ch, Q_ch


def gen_8psk_frame(seq_len: int, snr_db: float, normalize: bool = True):
    """
    8PSK:
    Eight points around the unit circle.
    """
    symbols = [
        (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4))
        for k in range(8)
    ]

    I_ch, Q_ch = [], []

    for _ in range(seq_len):
        si, sq = random.choice(symbols)

        i, q = add_awgn(si, sq, snr_db)

        I_ch.append(i)
        Q_ch.append(q)

    if normalize:
        return normalize_frame(I_ch, Q_ch)

    return I_ch, Q_ch


# ============================================================
# DATASET
# ============================================================

def build_dataset():
    """
    Builds or loads a synthetic dataset.
    Also stores the SNR value for each sample.
    """

    if cfg.FORCE_REGENERATE_DATA and cfg.CACHE.exists():
        cfg.CACHE.unlink()
        print("[dataset] Old cache deleted.")

    if cfg.CACHE.exists():
        print(f"[dataset] Loading cached dataset: {cfg.CACHE}")
        saved = torch.load(cfg.CACHE, map_location="cpu")
        return saved["X"], saved["y"], saved["snr"]

    print("[dataset] Generating new dataset...")

    snr_values = list(range(*cfg.SNR_RANGE_DB))

    generators = [
        gen_bpsk_frame,
        gen_qpsk_frame,
        gen_8psk_frame,
    ]

    X_list = []
    y_list = []
    snr_list = []

    per_snr = max(1, cfg.SAMPLES_PER_CLASS // len(snr_values))

    for label, gen_fn in enumerate(generators):
        for snr in snr_values:
            for _ in range(per_snr):
                I_ch, Q_ch = gen_fn(cfg.SEQ_LEN, float(snr), normalize=True)

                X_list.append([I_ch, Q_ch])
                y_list.append(label)
                snr_list.append(snr)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    snr = torch.tensor(snr_list, dtype=torch.long)

    torch.save(
        {
            "X": X,
            "y": y,
            "snr": snr,
        },
        cfg.CACHE,
    )

    print(f"[dataset] Saved dataset to: {cfg.CACHE}")
    print(f"[dataset] X shape: {X.shape}")
    print(f"[dataset] y shape: {y.shape}")

    return X, y, snr


def split_loaders_balanced(X, y, snr):
    """
    Balanced split by class and SNR.
    This prevents validation from having too many hard or easy samples.
    """

    groups = defaultdict(list)

    for idx, label in enumerate(y.tolist()):
        snr_value = int(snr[idx].item())
        groups[(label, snr_value)].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for _, indices in groups.items():
        random.shuffle(indices)

        n = len(indices)

        train_end = int(0.60 * n)
        val_end = int(0.80 * n)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    train_indices = torch.tensor(train_indices, dtype=torch.long)
    val_indices = torch.tensor(val_indices, dtype=torch.long)
    test_indices = torch.tensor(test_indices, dtype=torch.long)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]
    snr_test = snr[test_indices]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, X_test, y_test, snr_test


# ============================================================
# MODEL
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, drop: float = 0.0):
        super().__init__()

        padding = k // 2

        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=padding, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(ch, ch, k, padding=padding, bias=False),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class TripleSkipBlock(nn.Module):
    def __init__(self, ch: int, drop: float = 0.0):
        super().__init__()

        self.b1 = ResBlock(ch, drop=drop)
        self.b2 = ResBlock(ch, drop=drop)
        self.b3 = ResBlock(ch, drop=drop)

    def forward(self, x):
        return F.relu(self.b3(self.b2(self.b1(x))) + x)


class TRNN(nn.Module):
    def __init__(self):
        super().__init__()

        F_base = cfg.BASE_FILTERS
        D = cfg.DROPOUT

        self.stem = nn.Sequential(
            nn.Conv1d(2, F_base, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(F_base),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            TripleSkipBlock(F_base, D),
            nn.Conv1d(F_base, F_base * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(F_base * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.stage2 = nn.Sequential(
            TripleSkipBlock(F_base * 2, D),
            nn.Conv1d(F_base * 2, F_base * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(F_base * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.stage3 = nn.Sequential(
            TripleSkipBlock(F_base * 4, D),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F_base * 4, F_base * 2),
            nn.ReLU(),
            nn.Dropout(D),
            nn.Linear(F_base * 2, cfg.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)

        return x


# ============================================================
# TRAINING
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb = xb.to(cfg.DEVICE)
            yb = yb.to(cfg.DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print("\n[train] Training started")
    print(f"[train] Device: {cfg.DEVICE}")
    print(
        f"{'Epoch':>6} "
        f"{'Train Loss':>12} "
        f"{'Train Acc':>12} "
        f"{'Val Loss':>12} "
        f"{'Val Acc':>12}"
    )
    print("-" * 65)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"{epoch:6d} "
            f"{train_loss:12.4f} "
            f"{train_acc * 100:11.2f}% "
            f"{val_loss:12.4f} "
            f"{val_acc * 100:11.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(model.state_dict(), cfg.CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print("\n[train] Early stopping triggered.")
            break

    print(f"\n[train] Best validation loss: {best_val_loss:.4f}")
    print(f"[train] Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"[train] Best model saved to: {cfg.CHECKPOINT}")

    return history


# ============================================================
# EVALUATION
# ============================================================

def get_predictions(model, loader):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.DEVICE)

            logits = model(xb)
            pred = logits.argmax(1).cpu()

            preds.extend(pred.tolist())
            labels.extend(yb.tolist())

    return preds, labels


def evaluate_by_snr(preds, labels, snr_test):
    print("\nAccuracy by SNR:")
    print("-" * 35)

    snr_values = sorted(set(snr_test.tolist()))

    for snr_value in snr_values:
        idxs = [
            i for i, s in enumerate(snr_test.tolist())
            if s == snr_value
        ]

        correct = 0
        total = len(idxs)

        for i in idxs:
            if preds[i] == labels[i]:
                correct += 1

        acc = correct / total if total > 0 else 0.0

        print(f"SNR {snr_value:>3} dB: {acc * 100:6.2f}%")


# ============================================================
# PLOTS
# ============================================================

def plot_training(history):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        label="Train Accuracy",
    )
    plt.plot(
        epochs,
        [a * 100 for a in history["val_acc"]],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.OUT_DIR / "accuracy_curve.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.OUT_DIR / "loss_curve.png")
    plt.show()


def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1, 2], cfg.CLASSES)
    plt.yticks([0, 1, 2], cfg.CLASSES)

    for i in range(3):
        for j in range(3):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(cfg.OUT_DIR / "confusion_matrix.png")
    plt.show()


def plot_constellations_at_snr(snr_db: int, filename: str, axis_limit: float):
    """
    Plots constellation diagrams at a chosen SNR.
    Use 20 dB for clean constellations.
    Use 0 dB or -10 dB for noisy constellations.
    """

    seq_len = 1500

    generators = [
        gen_bpsk_frame,
        gen_qpsk_frame,
        gen_8psk_frame,
    ]

    plt.figure(figsize=(15, 5))

    for class_id, gen_fn in enumerate(generators):
        I_ch, Q_ch = gen_fn(
            seq_len,
            snr_db,
            normalize=True,
        )

        plt.subplot(1, 3, class_id + 1)
        plt.scatter(I_ch, Q_ch, s=8, alpha=0.5)

        plt.axhline(0, linewidth=0.7)
        plt.axvline(0, linewidth=0.7)

        plt.title(f"{cfg.CLASSES[class_id]} Constellation at {snr_db} dB")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-axis_limit, axis_limit)
        plt.ylim(-axis_limit, axis_limit)

    plt.tight_layout()
    plt.savefig(cfg.OUT_DIR / filename)
    plt.show()


def plot_clear_time_domain_signals():
    """
    Plots clean high-SNR time-domain I/Q signals.
    This is the best plot to use in your presentation.
    """

    snr_db = 20
    seq_len = 120

    generators = [
        gen_bpsk_frame,
        gen_qpsk_frame,
        gen_8psk_frame,
    ]

    plt.figure(figsize=(15, 8))

    for class_id, gen_fn in enumerate(generators):
        I_ch, Q_ch = gen_fn(
            seq_len,
            snr_db,
            normalize=True,
        )

        sample_index = list(range(seq_len))

        plt.subplot(3, 2, class_id * 2 + 1)
        plt.plot(sample_index, I_ch, linewidth=1.5)
        plt.axhline(0, linewidth=0.7)
        plt.title(f"{cfg.CLASSES[class_id]} Clean I Channel at {snr_db} dB")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(3, 2, class_id * 2 + 2)
        plt.plot(sample_index, Q_ch, linewidth=1.5)
        plt.axhline(0, linewidth=0.7)
        plt.title(f"{cfg.CLASSES[class_id]} Clean Q Channel at {snr_db} dB")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(cfg.OUT_DIR / "clear_time_domain_signals_20dB.png")
    plt.show()


def plot_noisy_time_domain_signals():
    """
    Plots noisy time-domain I/Q signals.
    This shows how noise affects the signal.
    """

    snr_db = 0
    seq_len = 120

    generators = [
        gen_bpsk_frame,
        gen_qpsk_frame,
        gen_8psk_frame,
    ]

    plt.figure(figsize=(15, 8))

    for class_id, gen_fn in enumerate(generators):
        I_ch, Q_ch = gen_fn(
            seq_len,
            snr_db,
            normalize=True,
        )

        sample_index = list(range(seq_len))

        plt.subplot(3, 2, class_id * 2 + 1)
        plt.plot(sample_index, I_ch, linewidth=1.5)
        plt.axhline(0, linewidth=0.7)
        plt.title(f"{cfg.CLASSES[class_id]} Noisy I Channel at {snr_db} dB")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(3, 2, class_id * 2 + 2)
        plt.plot(sample_index, Q_ch, linewidth=1.5)
        plt.axhline(0, linewidth=0.7)
        plt.title(f"{cfg.CLASSES[class_id]} Noisy Q Channel at {snr_db} dB")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(cfg.OUT_DIR / "noisy_time_domain_signals_0dB.png")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Improved TRNN Classifier: BPSK vs QPSK vs 8PSK")
    print("=" * 70)

    print(f"Device: {cfg.DEVICE}")

    X, y, snr = build_dataset()

    train_loader, val_loader, test_loader, X_test, y_test, snr_test = split_loaders_balanced(
        X,
        y,
        snr,
    )

    model = TRNN().to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n[model] Total parameters: {total_params:,}")
    print(f"[model] Input shape: batch, 2, {cfg.SEQ_LEN}")

    history = train(
        model,
        train_loader,
        val_loader,
    )

    print("\n[eval] Loading best model...")
    model.load_state_dict(
        torch.load(
            cfg.CHECKPOINT,
            map_location=cfg.DEVICE,
        )
    )

    preds, labels = get_predictions(
        model,
        test_loader,
    )

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=cfg.CLASSES,
        )
    )

    evaluate_by_snr(
        preds,
        labels,
        snr_test,
    )

    plot_training(history)
    plot_confusion_matrix(preds, labels)

    plot_constellations_at_snr(
        snr_db=20,
        filename="clean_constellations_20dB.png",
        axis_limit=1.7,
    )

    plot_constellations_at_snr(
        snr_db=0,
        filename="noisy_constellations_0dB.png",
        axis_limit=3.0,
    )

    plot_clear_time_domain_signals()
    plot_noisy_time_domain_signals()

    print(f"\nDone. All outputs saved to: {cfg.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()