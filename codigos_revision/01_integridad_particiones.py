# -*- coding: utf-8 -*-
"""
01_integridad_particiones.py — Auditoría de integridad de particiones (Sección 3.3; R1-1, contribución ii)
NUEVO (es el "data-integrity script" de Software S1).
Duplicados exactos (SHA-256), de píxeles decodificados (MD5 de RGB) y casi duplicados
perceptuales (pHash, distancia de Hamming <= 5) ENTRE particiones.
Uso: python 01_integridad_particiones.py cls     (dataset de clasificación, rutas de config.py)
     python 01_integridad_particiones.py det     (dataset de detección: <split>/images)
Salidas: tablas/T_M1_integrity_<cls|det>.csv (resumen) y listados de pares; NO modifica las particiones."""
import sys, hashlib, itertools, collections, pathlib
import numpy as np, pandas as pd
from PIL import Image
import imagehash
import config
RESULTS = config.TABLES

tag = sys.argv[1]
root = pathlib.Path(config.CLS_DIR if tag == "cls" else config.DET_DIR)
SPLITS = ["train", "valid", "test"]
VALID_ALIASES = ("valid", "val", "validation")   # nombre de la partición de validación según el dataset
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
rows = []
for sp in SPLITS:
    name = next((a for a in (VALID_ALIASES if sp == "valid" else (sp,)) if (root / a).exists()), None)
    if name is None:
        raise FileNotFoundError(f"No se encontró la partición '{sp}' en {root}")
    base = root / name / "images" if (root / name / "images").exists() else root / name
    for p in base.rglob("*"):
        if p.suffix.lower() not in EXT:
            continue
        raw = p.read_bytes()
        im = Image.open(p).convert("RGB")
        rows.append(dict(split=sp, path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                         pix=hashlib.md5(np.asarray(im).tobytes()).hexdigest(),
                         phash=str(imagehash.phash(im))))
df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/integrity_manifest_{tag}.csv", index=False)
print(df.groupby("split").size())

def cross(col):
    out = []
    for _, g in df.groupby(col):
        if g.split.nunique() > 1:
            for a, b in itertools.combinations(g.itertuples(), 2):
                if a.split != b.split:
                    out.append((a.split, b.split, a.path, b.path, 0))
    return out

def near(th=5):
    # Índice por 8 bloques de 8 bits: con distancia <= 7 al menos un bloque coincide.
    H = {r.path: imagehash.hex_to_hash(r.phash) for r in df.itertuples()}
    sp = dict(zip(df.path, df.split)); buckets = collections.defaultdict(list)
    for r in df.itertuples():
        for k in range(8):
            buckets[(k, r.phash[2 * k:2 * k + 2])].append(r.path)
    seen, out = set(), []
    for lst in buckets.values():
        for a, b in itertools.combinations(lst, 2):
            if sp[a] == sp[b] or (a, b) in seen:
                continue
            seen.add((a, b)); d = H[a] - H[b]
            if d <= th:
                out.append((sp[a], sp[b], a, b, int(d)))
    return out

summary = []
for kind, pairs in [("exact_sha256", cross("sha256")), ("decoded_pixels", cross("pix")),
                    ("perceptual_phash_le5", near(5))]:
    pd.DataFrame(pairs, columns=["split_a", "split_b", "path_a", "path_b", "hamming"]) \
      .to_csv(f"{RESULTS}/integrity_{tag}_{kind}.csv", index=False)
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        n = sum(1 for p in pairs if {p[0], p[1]} == {a, b})
        imgs_test = len({p[3] if p[1] == "test" else p[2] for p in pairs
                         if {p[0], p[1]} == {a, b} and "test" in (a, b)})
        summary.append(dict(dataset=tag, check=kind, splits=f"{a}-{b}", pairs=n,
                            test_images_involved=imgs_test if "test" in (a, b) else ""))
pd.DataFrame(summary).to_csv(f"{RESULTS}/T_M1_integrity_{tag}.csv", index=False)
print(pd.DataFrame(summary))
