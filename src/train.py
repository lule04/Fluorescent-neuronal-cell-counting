import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.notebook import tqdm

from src import config


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Weighted BCE loss  (Tashfeen eq. 1, Morelli & Clissa)
# ---------------------------------------------------------------------------

def weighted_bce_loss(pred, target, w_p=None, w_n=None):
    if w_p is None:
        w_p = config.W_POSITIVE
    if w_n is None:
        w_n = config.W_NEGATIVE
    eps = 1e-7
    loss = -w_p * target * torch.log(pred + eps) \
           -w_n * (1 - target) * torch.log(1 - pred + eps)
    return loss.mean()


# ---------------------------------------------------------------------------
# Trening petlja
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=None,
    lr=None,
    device=None,
):
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.LR_PATIENCE,
        factor=config.LR_FACTOR,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "epoch": [],
        "lr": [],
    }

    pbar = tqdm(total=num_epochs, desc="Trening")

    for epoch in range(num_epochs):
        # ---- train faza ----
        model.train()
        running_loss = 0.0
        n_samples = 0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            pred = model(images)
            loss = weighted_bce_loss(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        train_loss = running_loss / n_samples

        # ---- val faza ----
        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                pred = model(images)
                loss = weighted_bce_loss(pred, masks)
                val_running += loss.item() * images.size(0)
                val_n += images.size(0)

        val_loss = val_running / val_n

        # scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # logovanje
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["epoch"].append(epoch)
        metrics["lr"].append(current_lr)

        pbar.set_postfix({
            "train": f"{train_loss:.4f}",
            "val": f"{val_loss:.4f}",
            "lr": f"{current_lr:.2e}",
            "best_ep": best_epoch,
        })
        pbar.update(1)

        # early stopping
        if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping na epohi {epoch} "
                  f"(best = ep {best_epoch}, val_loss = {best_val_loss:.4f})")
            break

    pbar.close()

    # učitaj best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Najbolja epoha: {best_epoch}, val_loss: {best_val_loss:.4f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Threshold tuning na validaciji  (piksel-level F1 kao proxy)
# ---------------------------------------------------------------------------

def _pixel_f1(pred_binary, gt_binary):
    tp = (pred_binary * gt_binary).sum()
    fp = (pred_binary * (1 - gt_binary)).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def find_best_threshold(model, val_loader, device=None,
                        thresholds=None):
    if device is None:
        device = get_device()
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    model.eval()
    # sakupi sve predikcije i GT maske
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            pred = model(images).cpu()
            all_preds.append(pred)
            all_gts.append(masks)

    all_preds = torch.cat(all_preds)
    all_gts = torch.cat(all_gts)

    best_thresh, best_f1 = 0.5, 0.0
    results = []
    for t in thresholds:
        binary = (all_preds > t).float()
        f1 = _pixel_f1(binary, all_gts)
        results.append((t, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Najbolji threshold: {best_thresh:.2f} (piksel F1 = {best_f1:.4f})")
    return best_thresh, results


# ---------------------------------------------------------------------------
# Čuvanje predikcija i modela
# ---------------------------------------------------------------------------

def save_model(model, model_name, threshold):
    save_dir = config.MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f"{model_name}.pth")
    (save_dir / f"{model_name}_threshold.txt").write_text(f"{threshold:.4f}\n")
    print(f"Model sačuvan: {save_dir / model_name}.pth (threshold={threshold:.4f})")


def save_predictions(model, test_loader, model_name, device=None):
    if device is None:
        device = get_device()
    pred_dir = config.PREDS_DIR / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, _, image_ids in tqdm(test_loader, desc="Predikcije"):
            images = images.to(device)
            preds = model(images).cpu().numpy()
            for i, iid in enumerate(image_ids):
                heatmap = preds[i, 0]  # (H, W) float32 [0,1]
                np.save(pred_dir / f"{iid}.npy", heatmap)

    print(f"Predikcije sačuvane u {pred_dir} ({len(test_loader.dataset)} slika)")


# ---------------------------------------------------------------------------
# Pomoćna: podskup train ID-jeva za proširenje #2
# ---------------------------------------------------------------------------

def get_subset_ids(train_ids, fraction, seed=config.SEED):
    rng = np.random.RandomState(seed)
    n = max(1, int(len(train_ids) * fraction))
    indices = rng.choice(len(train_ids), size=n, replace=False)
    return [train_ids[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Broj parametara
# ---------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    for name, layer in model.named_children():
        n = sum(p.numel() for p in layer.parameters())
        print(f"  {name}: {n:,} parametara")
    print(f"  UKUPNO: {total:,}")
    return total
