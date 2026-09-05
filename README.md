# Brojanje fluorescentnih neuronskih ćelija

Projekat iz predmeta Mašinsko učenje na temu ,,Brojanje ćelija na fluorescentnim mikroskopskim slikama pomoću konvolucionih neuronskih mreža".

## O projektu

Cilj projekta je segmentacija i brojanje fluorescentnih neuronskih ćelija na mikroskopskim slikama koristeći c-ResUnet arhitekturu sa i bez attention gate-ova 

Trenirana su tri modela:
- **M0** — c-ResUnet baseline (ELU aktivacija, bez attention-a)
- **M1** — c-ResUnet + Attention Gates (ELU)
- **M2** — c-ResUnet + Attention Gates (ReLU)

## Skup podataka

FluoCells dataset (Clissa et al., 2021) — 283 fluorescentne mikroskopske slike dimenzija 1600×1200 px sa ručno anotiranim binarnim maskama.

Preuzimanje: https://amsacta.unibo.it/id/eprint/6706/

## Struktura

```
src/
    config.py          — putanje i hiperparametri
    data_loading.py    — učitavanje i augmentacija
    models.py          — c-ResUnet arhitektura
    train.py           — trening petlja
    postprocess.py     — binarizacija, watershed, brojanje
    evaluation.py      — centroid matching, metrike
notebooks/
    00-02              — upoznavanje, EDA, split
    03-04              — trening modela
    05                 — post-processing demo
    06                 — evaluacija i poređenje
    07                 — finalna demo sveska
models/                — sačuvane težine (.pth)
data/splits/           — zamrznuti train/val/test ID-jevi
```

## Literatura

1. Morelli, R., Clissa, L. et al. (2021). *Automating cell counting in fluorescent microscopy through deep learning with c-ResUnet.* Scientific Reports, 11(1), 22920.
2. Clissa, L. et al. (2021). *FluoCells dataset* [Data set]. AMS Acta, University of Bologna. DOI: 10.6092/unibo/amsacta/6706
3. Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* arXiv:1804.03999.
4. Tashfeen, A. (2024). *Enhancing Fluorescent Neuronal Cell Counting: c-ResUnet with Attention Gates.* (tema projekta)

## Tim

- Ana Velimirović
- Tamara Baranin
- Luka Đekić

Predmet: Mašinsko učenje, Matematički fakultet, Univerzitet u Beogradu, 2026

Profesor: Mladen Nikolić

Asistenti: Ognjen Milinković, Lucija Miličić
