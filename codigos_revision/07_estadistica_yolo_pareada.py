# -*- coding: utf-8 -*-
"""
07_estadistica_yolo_pareada.py — Inferencia PAREADA entre detectores (NUEVO; Sección 4.4, Tabla 9)
- Reconstruye AP por clase con emparejamiento tipo COCO (IoU >= t) e interpolación de 101 puntos.
- Bootstrap pareado (mismas imágenes remuestreadas para todos los modelos): IC 95 % de mAP@50,
  mAP@50-95 y de la diferencia ΔmAP entre pares; TOST con márgenes de ±1 y ±2 pp.
Entradas: imágenes y etiquetas YOLO del test (config.py) y predicciones/det_<Modelo>.json (script 06).
Salidas:  tablas/T_G_detection_bootstrap_CI.csv  (mAP con IC 95 % por modelo)
          tablas/T_I_detection_paired_delta.csv  (ΔmAP pareado, Holm y equivalencia ±1/±2 pp)
Uso: python 07_estadistica_yolo_pareada.py --boot 1000      (≈ 15-20 min con 6 detectores)
Nota: el mAP reconstruido aquí debe diferir < 0,5 pp del de Ultralytics (script 06); si no, avisar."""
import argparse, glob, json, os, itertools, pathlib
import numpy as np, pandas as pd
from PIL import Image
import config
from config import SEED
from estadistica import holm
PREDS, RESULTS = config.PREDS, config.TABLES

ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--test_dir", default=os.path.dirname(config.DET_TEST_IMAGES)); A = ap.parse_args()
THR = np.round(np.arange(0.50, 0.96, 0.05), 2)

def iou(b, G):  # b: [x1,y1,x2,y2]; G: (k,4)
    x1 = np.maximum(b[0], G[:, 0]); y1 = np.maximum(b[1], G[:, 1])
    x2 = np.minimum(b[2], G[:, 2]); y2 = np.minimum(b[3], G[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - inter)

# ---- Verdad de terreno ----
imgs = sorted(p for p in pathlib.Path(A.test_dir, "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
GT = {}
for p in imgs:
    W, H = Image.open(p).size; boxes = []
    lab = pathlib.Path(A.test_dir, "labels", p.stem + ".txt")
    if lab.exists():
        for line in open(lab):
            c, xc, yc, w, h = map(float, line.split()[:5])
            boxes.append((int(c), (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    GT[p.stem] = boxes
ids = list(GT); C = sorted({b[0] for v in GT.values() for b in v})

def per_image_records(pred_file):
    """Para cada imagen y clase: n_gt y lista (score, tp en cada umbral)."""
    P = {}
    dets = json.load(open(pred_file))
    # Ultralytics escribe category_id desde 1 (COCO) en datasets propios; las etiquetas YOLO empiezan en 0.
    cats = {int(d["category_id"]) for d in dets}
    off = 1 if (cats and 0 not in cats and max(cats) == max(C) + 1) else 0
    for d in dets:
        x, y, w, h = d["bbox"]
        P.setdefault(str(d["image_id"]), []).append((int(d["category_id"]) - off, d["score"], x, y, x + w, y + h))
    rec = {}
    for i in ids:
        for c in C:
            g = np.array([b[1:] for b in GT[i] if b[0] == c]).reshape(-1, 4)
            pr = sorted([p for p in P.get(i, []) if p[0] == c], key=lambda z: -z[1])
            tps = np.zeros((len(pr), len(THR)), bool)
            for ti, t in enumerate(THR):
                used = np.zeros(len(g), bool)
                for k, p in enumerate(pr):
                    if len(g) == 0: break
                    o = iou(np.array(p[2:]), g); o[used] = -1; j = int(o.argmax())
                    if o[j] >= t: used[j] = True; tps[k, ti] = True
            rec[(i, c)] = (len(g), np.array([p[1] for p in pr]), tps)
    return rec

def ap101(scores, tps, ngt):
    if ngt == 0: return np.nan
    o = np.argsort(-scores, kind="mergesort"); tp = tps[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / ngt, ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    rs = np.linspace(0, 1, 101); idx = np.searchsorted(rec, rs, side="left")
    return np.mean([prec[i] if i < len(prec) else 0 for i in idx])

def maps(rec, sample):
    """mAP@50 y mAP@50-95 para una muestra de imágenes (con repeticiones)."""
    res = np.zeros((len(C), len(THR)))
    for ci, c in enumerate(C):
        ngt = sum(rec[(i, c)][0] for i in sample)
        sc = np.concatenate([rec[(i, c)][1] for i in sample]) if sample else np.array([])
        tp = np.concatenate([rec[(i, c)][2] for i in sample]) if sample else np.zeros((0, len(THR)))
        for ti in range(len(THR)):
            res[ci, ti] = ap101(sc, tp[:, ti], ngt)
    return 100 * np.nanmean(res[:, 0]), 100 * np.nanmean(res)

models = {os.path.basename(f)[4:-5]: f for f in sorted(glob.glob(os.path.join(PREDS, "det_*.json")))}
R = {m: per_image_records(f) for m, f in models.items()}
point = {m: maps(R[m], ids) for m in models}
rng = np.random.default_rng(SEED); B = {m: [] for m in models}
for b in range(A.boot):
    s = list(rng.choice(ids, len(ids), replace=True))
    for m in models:
        B[m].append(maps(R[m], s))
B = {m: np.array(v) for m, v in B.items()}

rows = [{"Model": m, "mAP@50 %": round(point[m][0], 2),
         "mAP@50 95% CI": f"{np.percentile(B[m][:, 0], 2.5):.2f}–{np.percentile(B[m][:, 0], 97.5):.2f}",
         "mAP@50–95 %": round(point[m][1], 2),
         "mAP@50–95 95% CI": f"{np.percentile(B[m][:, 1], 2.5):.2f}–{np.percentile(B[m][:, 1], 97.5):.2f}"}
        for m in models]
pd.DataFrame(rows).to_csv(f"{RESULTS}/T_G_detection_bootstrap_CI.csv", index=False); print(pd.DataFrame(rows))

rows = []
for a, b in itertools.combinations(models, 2):
    for k, name in [(0, "mAP@50"), (1, "mAP@50–95")]:
        d = B[a][:, k] - B[b][:, k]; dp = point[a][k] - point[b][k]
        p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))   # p bilateral bootstrap
        lo90, hi90 = np.percentile(d, [5, 95])
        rows.append({"Comparison": f"{a} vs. {b}", "Metric": name, "Δ (pp)": round(dp, 2),
                     "95% CI (pp)": f"{np.percentile(d, 2.5):.2f} to {np.percentile(d, 97.5):.2f}",
                     "Bootstrap p": p_boot, "90% CI (pp)": f"{lo90:.2f} to {hi90:.2f}",
                     "Equivalent ±1 pp": "Yes" if (lo90 > -1 and hi90 < 1) else "No",
                     "Equivalent ±2 pp": "Yes" if (lo90 > -2 and hi90 < 2) else "No"})
T = pd.DataFrame(rows)
for name in T.Metric.unique():
    msk = T.Metric == name; T.loc[msk, "Holm-adjusted p"] = holm(T.loc[msk, "Bootstrap p"])
T.to_csv(f"{RESULTS}/T_I_detection_paired_delta.csv", index=False); print(T)
