from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Putanje do podataka
DATA_DIR = ROOT / "data" / "raw" / "fluocells"
IMG_DIR = DATA_DIR / "all_images" / "images"
MASK_DIR = DATA_DIR / "all_masks" / "masks"
SPLITS_DIR = ROOT / "data" / "splits"

SEED = 42

# Geometrija isecanja na 512x512
CROP_SIZE = 512
CROP_STARTS_X = [0, 288, 688, 1088]
CROP_STARTS_Y = [0, 288, 688]

# Filtriranje šuma
MIN_OBJECT_SIZE = 90
