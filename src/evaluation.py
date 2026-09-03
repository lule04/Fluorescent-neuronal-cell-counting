"""
Deo C — evaluacija: centroid matching (GT vs predikcija) i sve metrike.

Kontrakt (HANDOFF.md §8, DNEVNIK_ODLUKA "Predaja C-u"):
    evaluate(pred_masks, gt_masks, dist_thresh) -> dict(F1, MAE, MedAE, MPE, P, R, Acc, ...)

`pred_masks` / `gt_masks`: {image_id: 2D array (H,W)} — binarna (bool/0-1) ili već
labeled (int, 0=pozadina, 1..N=instance) maska. Funkcija sama radi centroid
matching preko `count_from_mask` iz `src.postprocess` ako dobije binarnu masku,
ili direktno čita centroide preko `regionprops` ako je maska već labeled.

Isti modul, isti prag rastojanja, korišćen za SVE modele — inače poređenje
nije validno (DNEVNIK_ODLUKA #5 iz evaluacionog protokola u HANDOFF §4).
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from src.postprocess import count_from_mask, _bbox_center

# Prag rastojanja za centroid matching (px). Original [1] Morelli koristi 50,
# tema-rad [Tema] Tashfeen koristi 40. Biramo 40: (a) direktno pratimo [Tema]
# protokol koji ceo projekat replicira/proširuje, (b) red veličine se slaže sa
# prosečnim prečnikom ćelije iz EDA (equivalent_diameter ~ 36.5 px) — "predikcija
# je tačna ako joj je centar unutar ~jednog prečnika ćelije od GT centra".
DEFAULT_DIST_THRESH = 40


def _centroids_from_mask(mask):
    """Vrati listu (row,col) centroida. Prihvata binarnu ILI već-labeled masku."""
    mask = np.asarray(mask)
    if mask.dtype == bool or set(np.unique(mask)).issubset({0, 1}):
        # binarna maska -> pun C-ov pipeline (clean+watershed) da dobijemo instance
        _, centroids, _ = count_from_mask(mask)
        return centroids
    # već labeled (npr. instance maska iz watershed-a) -> samo pročitaj regione
    # centar = bounding box centar (ISTA definicija kao count_from_mask, [Tema] §2, [6])
    return [_bbox_center(r) for r in regionprops(mask.astype(np.int32))]


def match_centroids(gt_centroids, pred_centroids, dist_thresh=DEFAULT_DIST_THRESH):
    """
    Optimalno bipartitno poklapanje (Hungarian algoritam) između GT i predviđenih
    centroida, ograničeno pragom rastojanja. Robustnije od pohlepnog
    "nearest-neighbor" pristupa: sprečava da dva predviđena objekta "otmu" istu
    GT ćeliju na način koji zavisi od redosleda obrade.

    Vraća: (tp, fp, fn, matches) gde je matches lista (gt_idx, pred_idx, dist).
    """
    n_gt, n_pred = len(gt_centroids), len(pred_centroids)
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0, []
    if n_gt == 0:
        return 0, n_pred, 0, []
    if n_pred == 0:
        return 0, 0, n_gt, []

    dist = cdist(np.asarray(gt_centroids), np.asarray(pred_centroids))
    row_ind, col_ind = linear_sum_assignment(dist)

    matches = []
    matched_gt, matched_pred = set(), set()
    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= dist_thresh:
            matches.append((int(r), int(c), float(dist[r, c])))
            matched_gt.add(r)
            matched_pred.add(c)

    tp = len(matches)
    fn = n_gt - len(matched_gt)
    fp = n_pred - len(matched_pred)
    return tp, fp, fn, matches


def evaluate(pred_masks, gt_masks, dist_thresh=DEFAULT_DIST_THRESH):
    """
    pred_masks, gt_masks: dict {image_id: 2D array}, isti ključevi (image_id).
    Vraća dict sa agregiranim metrikama + 'per_image' DataFrame (za dalju analizu/plotove).

    Detekcija (F1/Precision/Recall/Accuracy): TP/FP/FN SUMIRANI preko svih slika
    ("micro" agregacija — isto kao u [1]/[Tema]), ne prosek po slici, jer prosek
    po slici nije definisan za slike bez GT ćelija.

    Brojanje (MAE/MedAE/MPE): po slici, pa agregirano. MPE isključuje slike sa
    n_true=0 (deljenje nulom u formuli % greške — eksplicitno dokumentovana odluka,
    vidi DNEVNIK_ODLUKA).
    """
    image_ids = sorted(set(gt_masks) & set(pred_masks))
    missing_gt = set(pred_masks) - set(gt_masks)
    missing_pred = set(gt_masks) - set(pred_masks)
    if missing_gt or missing_pred:
        raise ValueError(
            f"pred_masks i gt_masks moraju imati iste image_id ključeve. "
            f"Nedostaje u gt: {missing_gt}, nedostaje u pred: {missing_pred}"
        )

    rows = []
    total_tp = total_fp = total_fn = 0

    for iid in image_ids:
        gt_centroids = _centroids_from_mask(gt_masks[iid])
        pred_centroids = _centroids_from_mask(pred_masks[iid])

        tp, fp, fn, _ = match_centroids(gt_centroids, pred_centroids, dist_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        n_true = len(gt_centroids)
        n_pred = len(pred_centroids)
        rows.append({
            "image_id": iid,
            "n_true": n_true,
            "n_pred": n_pred,
            "tp": tp, "fp": fp, "fn": fn,
            "abs_error": abs(n_true - n_pred),
        })

    per_image = pd.DataFrame(rows)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    mae = per_image["abs_error"].mean()
    medae = per_image["abs_error"].median()

    nonempty = per_image[per_image["n_true"] > 0]
    n_excluded_mpe = len(per_image) - len(nonempty)
    if len(nonempty) > 0:
        pct_errors = (nonempty["n_true"] - nonempty["n_pred"]) / nonempty["n_true"]
        mpe = 100.0 * pct_errors.mean()
    else:
        mpe = float("nan")

    return {
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        # kratki alias-i P/R/Acc — tačno kako je potpis definisan u HANDOFF.md §8
        # ("dict(F1, MAE, MedAE, MPE, P, R, Acc)"), pored punih imena radi čitljivosti
        "P": precision,
        "R": recall,
        "Acc": accuracy,
        "MAE": mae,
        "MedAE": medae,
        "MPE": mpe,
        "MPE_n_excluded_empty_gt": n_excluded_mpe,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "n_images": len(image_ids),
        "dist_thresh": dist_thresh,
        "per_image": per_image,
    }
