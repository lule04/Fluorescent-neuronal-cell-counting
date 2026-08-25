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

# === Deo B: trening ===
MODELS_DIR = ROOT / "models"
PREDS_DIR = ROOT / "preds"

LEARNING_RATE = 1e-4
BATCH_SIZE_BASELINE = 8
BATCH_SIZE_ATTENTION = 4
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
LR_PATIENCE = 4
LR_FACTOR = 0.7

W_POSITIVE = 1.5
W_NEGATIVE = 0.5

# Proširenje #2: frakcije train skupa za krivu podataka
DATA_FRACTIONS = [0.25, 0.50, 0.75, 1.0]
