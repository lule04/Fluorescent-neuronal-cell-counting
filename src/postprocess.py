"""
Post-processing: heatmap verovatnoća -> instance maska -> broj ćelija.

Pipeline (isti za GT i za predikcije, isti za sve modele):
    binarizuj (>threshold, ili >0 za GT) -> ukloni objekte < MIN_OBJECT_SIZE
    -> popuni rupe -> distance transform -> lokalni maksimumi (markeri)
    -> watershed (razdvaja dodirnute/zbijene ćelije) -> connected components
    -> centroidi = koordinate ćelija.

`count_from_mask()` je deljena funkcija — koristi se identično za GT i za
predikcije, što omogućava fer poređenje.

Centar ćelije se računa kao centar bounding box-a (sredina pravougaonika oko
ćelije). Watershed se primenjuje simetrično na GT i predikcije, što znači
da se oba izvora obrađuju istim pipeline-om — to je važno jer GT maske mogu
sadržavati stvarno dodirnute ćelije, a prost connected-components bi ih brojao
kao jedan objekat.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

from src import config

# Heuristika: markeri za watershed treba da budu razmaknuti ~poluprečnik tipične
# ćelije, inače se dodirnute ćelije istog tipa neće razdvojiti (premali min_distance)
# ili će se ista ćelija razbiti na više lažnih markera (preveliki min_distance).
# Iz EDA (notebooks/01_eda.ipynb): prosečna minor_axis_length ~29-33 px -> poluprečnik ~15px.
DEFAULT_MIN_DISTANCE = 15


def clean_binary_mask(binary_mask, min_size=None):
    """Ukloni sitne objekte (šum) i popuni rupe. Ulaz/izlaz: bool (H,W)."""
    if min_size is None:
        min_size = config.MIN_OBJECT_SIZE
    mask = np.asarray(binary_mask, dtype=bool)
    if not mask.any():
        return mask
    mask = remove_small_objects(mask, min_size=min_size)
    mask = binary_fill_holes(mask)
    return mask


def watershed_instances(clean_mask, min_distance=DEFAULT_MIN_DISTANCE):
    """
    Razdvoji dodirnute/zbijene objekte u already-cleaned binarnoj maski pomoću
    watershed-a nad distance transformom. Vraća labeled masku (int32, 0=pozadina,
    1..N = instance ćelija).
    """
    clean_mask = np.asarray(clean_mask, dtype=bool)
    if not clean_mask.any():
        return np.zeros(clean_mask.shape, dtype=np.int32)

    distance = distance_transform_edt(clean_mask)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=clean_mask,
        exclude_border=False,
    )
    if len(coords) == 0:
        # nema jasnih pikova (npr. objekat premali za distance transform) ->
        # tretiraj ceo mask kao jednu instancu preko connected components
        return label(clean_mask).astype(np.int32)

    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labeled = watershed(-distance, markers, mask=clean_mask)
    return labeled.astype(np.int32)


def _bbox_center(region):
    """
    Centar bounding box-a regiona (NE centroid/težište piksela). Za skoro-okrugle
    ćelije se poklapa sa centroidom; razlika se javlja kod izduženih/nepravilnih
    (npr. posle watershed-a) oblika.
    """
    min_row, min_col, max_row, max_col = region.bbox
    return ((min_row + max_row) / 2, (min_col + max_col) / 2)


def count_from_mask(binary_mask, min_size=None, min_distance=DEFAULT_MIN_DISTANCE):
    """
    Kanonsko brojanje ćelija iz VEĆ BINARNE maske (GT ili predikcija posle praga).
    Ista funkcija za GT i predikcije osigurava fer poređenje.

    Vraća: (labeled_mask (int32 HxW), centroids (lista (row,col)), count (int))
    """
    clean = clean_binary_mask(binary_mask, min_size=min_size)
    labeled = watershed_instances(clean, min_distance=min_distance)
    regions = regionprops(labeled)
    centroids = [_bbox_center(r) for r in regions]
    return labeled, centroids, len(regions)


def predict_mask(heatmap, threshold, min_size=None, min_distance=DEFAULT_MIN_DISTANCE):
    """
    Pun pipeline OD heatmap-a (model izlaz [0,1]) DO instanci ćelija.
    threshold: prag binarizacije (iz models/{model}_threshold.txt, biran na validaciji).
    """
    binary = np.asarray(heatmap) > threshold
    return count_from_mask(binary, min_size=min_size, min_distance=min_distance)
