import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config


def _is_empty(iid):
    # prazna maska = nema nijednog piksela >0 (ista binarizacija kao u EDA/data_loading)
    return int((np.array(Image.open(config.MASK_DIR / iid)) > 0).sum()) == 0


def make_splits(test_ids, seed=config.SEED, val_frac=0.2):
    all_ids = sorted(p.name for p in config.IMG_DIR.glob("*.png"))
    assert set(test_ids) <= set(all_ids) and len(test_ids) == 70, "test lista nije ispravna"
    remaining = [i for i in all_ids if i not in set(test_ids)]
    empt = [_is_empty(i) for i in remaining]
    train, val = train_test_split(remaining, test_size=val_frac, random_state=seed, stratify=empt)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test_ids)}


def save_splits(splits):
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (config.SPLITS_DIR / f"{name}_image_ids.txt").write_text("\n".join(ids) + "\n")
