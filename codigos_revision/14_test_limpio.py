# -*- coding: utf-8 -*-
"""
14_test_limpio.py — Análisis del test SIN casi-duplicados (respuesta a la auditoría del paso 6; R1-1, R1-3)
NUEVO. La auditoría (01_integridad_particiones.py) marcó como casi-duplicados (pHash, Hamming <= 5):
  clasificación: 240 de 562 imágenes de prueba con pareja en entrenamiento y 100 con pareja en validación;
  detección:     10 de 757 con pareja en entrenamiento y 143 con pareja en validación.
Este script:
  1) genera una muestra visual de pares marcados (para confirmar si son la misma fruta) y una plantilla CSV
     para anotar la revisión visual (columna same_fruit: si / no / dudoso);
  2) recalcula los resultados excluyendo las imágenes de prueba marcadas:
       - variante A: sin las marcadas con pareja en ENTRENAMIENTO
       - variante B: sin las marcadas con pareja en entrenamiento O validación
     clasificación: exactitud (IC de Wilson), F1 macro, McNemar exacto + Holm entre los CNN;
     detección: mAP@50 y mAP@50-95 reconstruidos (mismo método del script 07).
Requiere: tablas/integrity_cls_perceptual_phash_le5.csv, tablas/integrity_det_perceptual_phash_le5.csv,
          predicciones/cls_*.csv y predicciones/det_*.json (scripts 01, 04 y 06).
Salidas:  tablas/T_N_clean_test_classification.csv, tablas/T_N_clean_test_mcnemar.csv,
          tablas/T_N_clean_test_detection.csv, tablas/T_N_hamming_distribution.csv,
          tablas/revision_visual_pares_<cls|det>.csv, figuras/near_duplicates_sample_<cls|det>.png
Uso: python 14_test_limpio.py --muestra 24
"""
import argparse, glob, json, os, itertools, pathlib, random
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import config
from estadistica import wilson, mcnemar_exact, holm, paired_counts

ap = argparse.ArgumentParser(); ap.add_argument("--muestra", type=int, default=24); A = ap.parse_args()
T = config.TABLES; TEST_DIR = os.path.dirname(config.DET_TEST_IMAGES)
rng = random.Random(config.SEED)

def load_pairs(tag):
    f = os.path.join(T, f"integrity_{tag}_perceptual_phash_le5.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Falta {f}: ejecutar antes 01_integridad_particiones.py {tag}")
    p = pd.read_csv(f)
    p = p[(p.split_a == "test") | (p.split_b == "test")].copy()
    p["test_path"] = np.where(p.split_a == "test", p.path_a, p.path_b)
    p["other_path"] = np.where(p.split_a == "test", p.path_b, p.path_a)
    p["other_split"] = np.where(p.split_a == "test", p.split_b, p.split_a)
    return p

# ---------------------------------------------------------------------------
# 1) Muestra visual y plantilla de revisión
# ---------------------------------------------------------------------------
hrows = []
for tag in ("cls", "det"):
    p = load_pairs(tag)
    for sp in ("train", "valid"):
        q = p[p.other_split == sp]
        hrows += [{"dataset": tag, "pair": f"test–{sp}", "Hamming": h, "pairs": int(n)} for h, n in q.hamming.value_counts().sort_index().items()]
    best = p.sort_values("hamming").drop_duplicates("test_path")          # pareja más parecida por imagen de prueba
    sample = best.sample(n=min(A.muestra, len(best)), random_state=config.SEED) if len(best) else best
    sample[["test_path", "other_path", "other_split", "hamming"]].assign(same_fruit="", comentario="") \
        .to_csv(os.path.join(T, f"revision_visual_pares_{tag}.csv"), index=False)
    if len(sample):
        n = len(sample); cols = 4; rows_ = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows_, cols * 2, figsize=(cols * 2 * 1.6, rows_ * 1.8))
        axs = np.atleast_2d(axs)
        for k, r in enumerate(sample.itertuples()):
            i, j = divmod(k, cols)
            for jj, (pth, lab) in enumerate([(r.test_path, "test"), (r.other_path, r.other_split)]):
                a = axs[i, 2 * j + jj]; a.imshow(Image.open(pth).convert("RGB")); a.axis("off")
                a.set_title(f"{lab} (H={r.hamming})" if jj == 0 else lab, fontsize=6)
        for a in axs.flat:
            a.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(config.FIGS, f"near_duplicates_sample_{tag}.png"), dpi=300); plt.close(fig)
pd.DataFrame(hrows).to_csv(os.path.join(T, "T_N_hamming_distribution.csv"), index=False)

# ---------------------------------------------------------------------------
# 2a) Clasificación sin las imágenes marcadas
# ---------------------------------------------------------------------------
pc = load_pairs("cls")
rel = lambda pth: os.path.relpath(pth, config.CLS_TEST_DIR).replace("\\", "/")
flag_train = {rel(x) for x in pc[pc.other_split == "train"].test_path}
flag_any = {rel(x) for x in pc.test_path}
P = {os.path.basename(f)[4:-4]: pd.read_csv(f).set_index("image_id") for f in sorted(glob.glob(os.path.join(config.PREDS, "cls_*.csv")))}
crow, mrow = [], []
for label, excl in [("Official test (all)", set()), ("A: without near-duplicates of training images", flag_train),
                    ("B: without near-duplicates of training or validation images", flag_any)]:
    ok = {}
    for m, d in P.items():
        dd = d[~d.index.isin(excl)]; k = int(dd.correct.sum()); acc, lo, hi = wilson(k, len(dd)); ok[m] = dd.correct
        crow.append({"Subset": label, "Model": m, "n": len(dd), "Correct": k, "Accuracy % (95% Wilson CI)": f"{acc:.2f} ({lo:.2f}–{hi:.2f})",
                     "Macro F1 %": round(100 * f1_score(dd.true_label, dd.pred_label, labels=config.CLASSES, average="macro", zero_division=0), 2)})
    orig = [m for m in ("MicroNet_Banana", "ResNet50V2", "Xception") if m in ok]
    rr = []
    for a, b in itertools.combinations(orig, 2):
        x, y = paired_counts(ok[a].loc[ok[b].index], ok[b]); rr.append({"Subset": label, "Comparison": f"{a} vs. {b}", "Discordant counts": f"{x} vs. {y}", "Raw p": mcnemar_exact(x, y)})
    if rr:
        R_ = pd.DataFrame(rr); R_["Holm-adjusted p"] = holm(R_["Raw p"]); mrow.append(R_)
pd.DataFrame(crow).to_csv(os.path.join(T, "T_N_clean_test_classification.csv"), index=False)
if mrow:
    pd.concat(mrow).to_csv(os.path.join(T, "T_N_clean_test_mcnemar.csv"), index=False)
print(pd.DataFrame(crow).to_string(index=False))

# ---------------------------------------------------------------------------
# 2b) Detección sin las imágenes marcadas (mAP reconstruido, igual que el script 07)
# ---------------------------------------------------------------------------
THR = np.round(np.arange(0.50, 0.96, 0.05), 2)

def iou(b, G):  # b: [x1,y1,x2,y2]; G: (k,4)
    x1 = np.maximum(b[0], G[:, 0]); y1 = np.maximum(b[1], G[:, 1])
    x2 = np.minimum(b[2], G[:, 2]); y2 = np.minimum(b[3], G[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]) - inter)

# ---- Verdad de terreno ----
imgs = sorted(p for p in pathlib.Path(TEST_DIR, "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
GT = {}
for p in imgs:
    W, H = Image.open(p).size; boxes = []
    lab = pathlib.Path(TEST_DIR, "labels", p.stem + ".txt")
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


pdet = load_pairs("det")
stem = lambda pth: pathlib.Path(pth).stem
dflag_train = {stem(x) for x in pdet[pdet.other_split == "train"].test_path}
dflag_any = {stem(x) for x in pdet.test_path}
models = {os.path.basename(f)[4:-5]: f for f in sorted(glob.glob(os.path.join(config.PREDS, "det_*.json")))}
drow = []
for m, f in models.items():
    R = per_image_records(f)
    for label, excl in [("Official test (all)", set()), ("A: without near-duplicates of training images", dflag_train),
                        ("B: without near-duplicates of training or validation images", dflag_any)]:
        s = [i for i in ids if i not in excl]; m50, m5095 = maps(R, s)
        drow.append({"Subset": label, "Model": m, "n images": len(s), "mAP@50 %": round(m50, 2), "mAP@50–95 %": round(m5095, 2)})
pd.DataFrame(drow).to_csv(os.path.join(T, "T_N_clean_test_detection.csv"), index=False)
print(pd.DataFrame(drow).to_string(index=False))
print("\nRevisar las figuras near_duplicates_sample_*.png y completar la columna same_fruit en revision_visual_pares_*.csv")
