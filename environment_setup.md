# Podešavanje okruženja

## Koraci

1. **Napraviti okruženje** iz `environment.yml` (micromamba ili conda):

   ```bash
   micromamba env create -f environment.yml
   # ili: conda env create -f environment.yml
   ```

   Ovo pravi okruženje po imenu `projekat` sa Python 3.11 i svim potrebnim
   bibliotekama (numpy, pandas, pillow, matplotlib, scipy, scikit-image,
   jupyterlab, ipykernel).

2. **Izabrati kernel** — u Jupyteru (ili VS Code/JupyterLab) otvoriti svesku i
   izabrati kernel **"projekat"** (odgovara okruženju iz koraka 1).

3. **Otvoriti sveske po redosledu** (00, 01, 02, ...) i pokrenuti **Run All**.
   Nije potreban nikakav dodatni `pip install` projekta — prva ćelija svake
   sveske sama pronalazi koren repoa (folder koji sadrži `src/`) i dodaje ga na
   `sys.path`, tako da `from src import config` radi bez obzira odakle je
   Jupyter pokrenut.

## Napomena o podacima

Dataset (`fluocells/`) se ne nalazi u repou, te se mora ručno postaviti u `data/raw/fluocells/` pre pokretanja
svesaka.
