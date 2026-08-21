import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from .config import IMG_DIR, MASK_DIR

def show_image_mask_and_overlay(name):
    img = np.array(Image.open(IMG_DIR/name))
    msk = np.array(Image.open(MASK_DIR/name)) > 0
    fig, ax = plt.subplots(1,3, figsize=(20,5))
    ax[0].imshow(img)
    ax[0].set_title(f"slika {name}")
    ax[1].imshow(msk, cmap="gray")
    ax[1].set_title("maska")
    ax[2].imshow(img)
    ax[2].imshow(msk, cmap="Reds", alpha=0.4); ax[2].set_title("preklop")
    for a in ax: a.axis("off")
    plt.show()