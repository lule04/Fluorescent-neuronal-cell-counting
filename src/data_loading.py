import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import config

# 12 fiksnih prozora (deterministički redosled)
CROP_WINDOWS = [(x, y) for x in config.CROP_STARTS_X for y in config.CROP_STARTS_Y]


def _train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, border_mode=0, p=0.5),   # rotacije; border_mode=0 -> popuna nulom
        A.RandomBrightnessContrast(p=0.5),           # promena osvetljenja
        A.GaussNoise(p=0.3),                         # Gausov šum
        A.ElasticTransform(p=0.3),                   # elastična deformacija
    ])


def _to_tensor():
    return A.Compose([
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),  # -> [0,1]
        ToTensorV2(),   # slika HWC->CHW; maska ostaje (H,W)
    ])


class CellDataset(Dataset):
    """
       Dvorežimski dataset: form='crops' (512x512) ili form='full' (cela slika).
       Vraća (image (3,H,W) float [0,1], mask (1,H,W) float {0,1}, image_id).
    """

    def __init__(self, image_ids, form="full", augment=False, cache=True):
        self.form = form
        self.aug = _train_aug() if augment else None
        self.to_tensor = _to_tensor()
        self._use_cache = cache
        self._img_cache, self._mask_cache = {}, {}
        if form == "crops":
            self.samples = [(iid, ci) for iid in image_ids for ci in range(len(CROP_WINDOWS))]
        else:
            self.samples = [(iid, None) for iid in image_ids]

    def _load_full(self, image_id):
        if self._use_cache and image_id in self._img_cache:
            return self._img_cache[image_id], self._mask_cache[image_id]
        img = np.array(Image.open(config.IMG_DIR / image_id).convert("RGB"))
        mask = (np.array(Image.open(config.MASK_DIR / image_id)) > 0).astype(np.uint8)
        if self._use_cache:
            self._img_cache[image_id] = img
            self._mask_cache[image_id] = mask
        return img, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iid, ci = self.samples[idx]
        img, mask = self._load_full(iid)
        if self.form == "crops":
            x, y = CROP_WINDOWS[ci]
            img = img[y:y + config.CROP_SIZE, x:x + config.CROP_SIZE]
            mask = mask[y:y + config.CROP_SIZE, x:x + config.CROP_SIZE]
        if self.aug is not None:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        out = self.to_tensor(image=img, mask=mask)
        image_t = out["image"].float()                # (3,H,W) [0,1]
        mask_t = out["mask"].unsqueeze(0).float()      # (1,H,W) {0,1}
        return image_t, mask_t, iid


def _read_split(split):
    path = config.SPLITS_DIR / f"{split}_image_ids.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def make_dataset(split=None, image_ids=None, form="full", augment=False, cache=True):
    """
    Vrati CellDataset. `split` čita zamrznuti fajl data/splits/{split}_image_ids.txt;
    alternativno prosledi `image_ids` listu (korisno za smoke test dok split ne postoji).
    cache=False isključuje keširanje celih slika u RAM (za mašine sa malo memorije).
    """
    if image_ids is None:
        image_ids = _read_split(split)
    return CellDataset(image_ids, form=form, augment=augment, cache=cache)
