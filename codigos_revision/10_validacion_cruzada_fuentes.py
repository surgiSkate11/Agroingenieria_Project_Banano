# -*- coding: utf-8 -*-
"""
10_validacion_cruzada_fuentes.py — Validación cruzada entre las dos fuentes (R1-3, R2-13; nueva Sección 4.6)
NUEVO. Responde a "¿por qué no evaluar ambos métodos con ambas fuentes?":
 (a) CLASIFICADORES sobre recortes de las cajas reales del test de detección (margen 10 %), 6 -> 4 clases
 (b) DETECTORES sobre las 562 imágenes del test de clasificación (detección de mayor confianza; sin detección = error)
 (c) McNemar exacto entre el mejor CNN y el mejor detector sobre las MISMAS 562 imágenes
Ningún modelo se reentrena. Revisar el mapeo DET_TO_CLS de config.py con ejemplos de cada clase.
Salidas: tablas/T_J_cross_source.csv, predicciones/cross_*.csv
"""
import os, glob
import config
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from estadistica import wilson, mcnemar_exact, paired_counts
from cargar_modelos import load_classifier, load_detector, top1_class

config.require_gpu()
CNNS = list(config.available(config.CNN_MODELS)); DETS = list(config.available(config.YOLO_MODELS))
orig_acc = {}
f = os.path.join(config.TABLES, "control_aciertos_cnn.csv")
if os.path.exists(f):
    orig_acc = dict(zip(*pd.read_csv(f)[["Modelo", "Exactitud %"]].T.values))
# nombres de clase del detector (para el mapeo)
names = load_detector(DETS[0]).names
rows = []

# ---- (a) recortes de cajas reales ----
crops, y = [], []
for img in config.list_images(config.DET_TEST_IMAGES):
    lab = os.path.join(config.DET_TEST_LABELS, os.path.splitext(os.path.basename(img))[0] + ".txt")
    if not os.path.exists(lab):
        continue
    im = Image.open(img).convert("RGB"); W, H = im.size
    for line in open(lab):
        c, xc, yc, w, h = map(float, line.split()[:5]); w, h = w * 1.1, h * 1.1
        crops.append(im.crop((max(0, (xc - w / 2) * W), max(0, (yc - h / 2) * H), min(W, (xc + w / 2) * W), min(H, (yc + h / 2) * H))))
        y.append(config.det_class(names[int(c)]))
y = np.array(y); print(f"[-] {len(y)} recortes; por clase:", pd.Series(y).value_counts().to_dict())
for name in CNNS:
    fn = load_classifier(name)
    P = np.concatenate([fn(crops[i:i + 64]) for i in range(0, len(crops), 64)])
    pred = np.array(config.CLASSES)[P.argmax(1)]; k = int((pred == y).sum()); acc, lo, hi = wilson(k, len(y))
    pd.DataFrame({"true_label": y, "pred_label": pred}).to_csv(os.path.join(config.PREDS, f"cross_cls_on_detcrops_{name}.csv"), index=False)
    rows.append({"Model": name, "Trained on": "Classification release", "Evaluated on": "Detection test crops (GT boxes)",
                 "n": len(y), "Accuracy % (95% CI)": f"{acc:.2f} ({lo:.2f}–{hi:.2f})",
                 "Macro F1 %": round(100 * f1_score(y, pred, labels=config.CLASSES, average="macro"), 2),
                 "Drop vs. original test (pp)": round(orig_acc.get(name, np.nan) - acc, 2)})

# ---- (b) detectores sobre el test de clasificación ----
paths = config.list_images(config.CLS_TEST_DIR)
ids = [os.path.relpath(p, config.CLS_TEST_DIR).replace("\\", "/") for p in paths]
ytrue = np.array([config.nice(os.path.basename(os.path.dirname(p))) for p in paths])
ok_det = {}
for name in DETS:
    m = load_detector(name)
    pred = np.array([top1_class(m, p) for p in paths]); ok = (pred == ytrue).astype(int); ok_det[name] = ok
    acc, lo, hi = wilson(int(ok.sum()), len(ok))
    pd.DataFrame({"image_id": ids, "true_label": ytrue, "pred_label": pred, "correct": ok}) \
      .to_csv(os.path.join(config.PREDS, f"cross_det_on_cls_{name}.csv"), index=False)
    rows.append({"Model": name, "Trained on": "Detection release", "Evaluated on": "Classification test (562 images)",
                 "n": len(ok), "Accuracy % (95% CI)": f"{acc:.2f} ({lo:.2f}–{hi:.2f})",
                 "Macro F1 %": round(100 * f1_score(ytrue, pred, labels=config.CLASSES, average="macro"), 2),
                 "Drop vs. original test (pp)": "", "No detection": int((pred == "none").sum())})
T = pd.DataFrame(rows); T.to_csv(os.path.join(config.TABLES, "T_J_cross_source.csv"), index=False); print(T.to_string(index=False))

# ---- (c) mejor CNN vs. mejor detector (mismas 562 imágenes) ----
best_cnn = max(CNNS, key=lambda n: pd.read_csv(os.path.join(config.PREDS, f"cls_{n}.csv")).correct.sum())
cnn_ok = pd.read_csv(os.path.join(config.PREDS, f"cls_{best_cnn}.csv")).set_index("image_id").loc[ids].correct.values
best_det = max(ok_det, key=lambda n: ok_det[n].sum())
b, c = paired_counts(cnn_ok, ok_det[best_det])
txt = f"McNemar exacto {best_cnn} vs {best_det} (562 imágenes): solo CNN = {b}, solo detector = {c}, p = {mcnemar_exact(b, c):.6f}"
print(txt); open(os.path.join(config.TABLES, "T_J_mcnemar_cnn_vs_detector.txt"), "w").write(txt + "\n")
