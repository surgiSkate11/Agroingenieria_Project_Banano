# -*- coding: utf-8 -*-
"""
11_validacion_externa.py — Validación EXTERNA con un dataset público independiente: BananaImageBD
(R1-3, R2-13; Secciones 3.9 y 4.6 del manuscrito)

Dataset: Ferdaus et al. (2025) BananaImageBD, Data in Brief 58, 111239. doi:10.1016/j.dib.2024.111239
         Datos: doi:10.17632/ptfscwtnyz.2 (Mendeley Data), licencia CC BY 4.0.
         Subconjunto de madurez: 820 imágenes originales (Green 212, Semi-ripe 204, Ripe 201, Overripe 203),
         iPhone 13 Pro Max, fondo blanco, luz LED, Bangladés, cuatro variedades locales.
Protocolo:
  * SOLO imágenes originales (no las aumentadas).
  * Ningún modelo se reentrena ni se ajusta: inferencia directa con los pesos finales.
  * Mapeo (config.EXT_MAP): Green -> Unripe, Ripe -> Ripe, Overripe -> Overripe.
    Semi-ripe no tiene equivalente y se analiza aparte (a qué clase la asigna cada modelo).
    BananaImageBD no tiene Rotten: una predicción "Rotten" cuenta como error.
  * Métricas en las 3 clases mapeadas: exactitud con IC de Wilson, recall por clase, F1 macro (3 clases).
  * Referencia comparable: exactitud del test oficial RESTRINGIDA a Unripe, Ripe y Overripe (mismo conjunto de clases).
  * McNemar exacto + Holm entre los tres CNN del manuscrito en las mismas imágenes externas.
  * Detectores: clase de la detección de mayor confianza (sin detección = error).
Salidas (tablas/):
  T_K0_external_description.csv   número de imágenes por estado y mapeo
  T_K_external_results.csv        resultados por modelo (amplía la nueva tabla de la Sección 4.6)
  T_K_external_classwise.csv      recall por clase
  T_K_semiripe_assignment.csv     a qué clase se asigna Semi-ripe
  T_K_external_mcnemar.csv        McNemar + Holm entre CNN en el conjunto externo
"""
import os, itertools, re
import config
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, recall_score
from estadistica import wilson, mcnemar_exact, holm, paired_counts
from cargar_modelos import load_classifier, load_detector, top1_class

config.require_gpu(); config.save_environment("externa")
if not os.path.isdir(config.EXT_DIR):
    raise FileNotFoundError(f"No existe EXT_DIR = {config.EXT_DIR}. Revisar la ruta en config.py.")

# ---------------------------------------------------------------------------
# 1. Lectura del dataset: una subcarpeta por estado de madurez
# ---------------------------------------------------------------------------
norm = lambda s: re.sub(r"[^a-z]", "", s.lower())           # "Semi-ripe" -> "semiripe"
rows = []
for p in config.list_images(config.EXT_DIR):
    parts = os.path.relpath(p, config.EXT_DIR).replace("\\", "/").split("/")
    if any("aug" in q.lower() for q in parts):                  # seguridad: excluir aumentadas (carpeta o archivo)
        continue
    # estado de madurez = la carpeta (más cercana al archivo) cuyo nombre reconocemos
    stage = next((q for q in reversed(parts[:-1]) if norm(q) in config.EXT_MAP), None)
    if stage is None:
        continue
    rows.append({"file": "/".join(parts), "path": p, "stage": stage, "true_label": config.EXT_MAP[norm(stage)]})
if not rows:
    raise FileNotFoundError(f"No se encontraron imágenes de Green/Semi-ripe/Ripe/Overripe bajo {config.EXT_DIR}")
meta = pd.DataFrame(rows)
meta["stage"] = meta["stage"].map(lambda s: s.strip())
desc = meta.groupby("stage").size().rename("n").to_frame()
desc["Mapped to"] = [config.EXT_MAP[norm(s)] or "— (analysed separately)" for s in desc.index]
desc.to_csv(os.path.join(config.TABLES, "T_K0_external_description.csv"))
print(desc, f"\nTotal: {len(meta)} imágenes (esperadas: 820 originales)")
if len(meta) > 900:
    print("[aviso] Hay más de 900 imágenes: verificar que EXT_DIR apunta a las ORIGINALES y no a las aumentadas.")

mapped = meta[meta.true_label.notna()].reset_index(drop=True)      # Green, Ripe, Overripe
semi = meta[meta.true_label.isna()].reset_index(drop=True)         # Semi-ripe
L3 = ["Unripe", "Ripe", "Overripe"]

# Referencia: exactitud en el test oficial restringido a las mismas 3 clases
def official_acc3(name):
    f = os.path.join(config.PREDS, f"cls_{name}.csv")
    if not os.path.exists(f):
        return np.nan
    d = pd.read_csv(f); d = d[d.true_label.isin(L3)]
    return 100 * d.correct.mean()

# ---------------------------------------------------------------------------
# 2. Inferencia
# ---------------------------------------------------------------------------
res, cw, sr, ok_cnn = [], [], [], {}
def evaluate(name, kind, pred_mapped, pred_semi):
    y = mapped.true_label.values; ok = (pred_mapped == y).astype(int)
    acc, lo, hi = wilson(int(ok.sum()), len(ok)); ref = official_acc3(name) if kind == "CNN" else np.nan
    res.append({"Model": name, "Type": kind, "n": len(ok), "Correct": int(ok.sum()),
                "Accuracy % (95% Wilson CI)": f"{acc:.2f} ({lo:.2f}–{hi:.2f})",
                "Macro F1 % (3 classes)": round(100 * f1_score(y, pred_mapped, labels=L3, average="macro", zero_division=0), 2),
                "Official test accuracy % (same 3 classes)": round(ref, 2),
                "Drop (pp)": round(ref - acc, 2) if not np.isnan(ref) else "",
                "Predicted as Rotten": int((pred_mapped == "Rotten").sum())})
    rec = recall_score(y, pred_mapped, labels=L3, average=None, zero_division=0)
    cw.append({"Model": name, **{f"Recall {c} %": round(100 * r, 2) for c, r in zip(L3, rec)}})
    vc = pd.Series(pred_semi).value_counts()
    sr.append({"Model": name, "Semi-ripe n": len(pred_semi),
               **{f"→ {c} %": round(100 * vc.get(c, 0) / max(len(pred_semi), 1), 2) for c in config.CLASSES + ["none"]}})
    pd.DataFrame({"file": list(mapped.file) + list(semi.file), "stage": list(mapped.stage) + list(semi.stage),
                  "true_label": list(mapped.true_label) + [None] * len(semi),
                  "pred_label": list(pred_mapped) + list(pred_semi)}) \
      .to_csv(os.path.join(config.PREDS, f"external_{name}.csv"), index=False)
    return ok

def batched(fn, paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        out.append(fn([Image.open(p).convert("RGB") for p in paths[i:i + bs]]))
    return np.concatenate(out) if out else np.zeros((0, 4))

for name in config.available(config.CNN_MODELS):
    print(f">>> {name}")
    fn = load_classifier(name)
    pm = np.array(config.CLASSES)[batched(fn, list(mapped.path)).argmax(1)]
    ps = np.array(config.CLASSES)[batched(fn, list(semi.path)).argmax(1)] if len(semi) else np.array([])
    ok_cnn[name] = evaluate(name, "CNN", pm, ps)
for name in config.available(config.YOLO_MODELS):
    print(f">>> {name}")
    m = load_detector(name)
    pm = np.array([top1_class(m, p) for p in mapped.path]); ps = np.array([top1_class(m, p) for p in semi.path])
    evaluate(name, "Detector (top-1)", pm, ps)

T = pd.DataFrame(res); T.to_csv(os.path.join(config.TABLES, "T_K_external_results.csv"), index=False)
pd.DataFrame(cw).to_csv(os.path.join(config.TABLES, "T_K_external_classwise.csv"), index=False)
pd.DataFrame(sr).to_csv(os.path.join(config.TABLES, "T_K_semiripe_assignment.csv"), index=False)
print(T.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. McNemar exacto + Holm entre los CNN del manuscrito en el conjunto externo
# ---------------------------------------------------------------------------
orig = [n for n in ("MicroNet_Banana", "ResNet50V2", "Xception") if n in ok_cnn]
rows = []
for a, b in itertools.combinations(orig, 2):
    x, y = paired_counts(ok_cnn[a], ok_cnn[b])
    rows.append({"Comparison": f"{a} vs. {b}", "Discordant counts": f"{x} vs. {y}", "Raw p": mcnemar_exact(x, y)})
if rows:
    M = pd.DataFrame(rows); M["Holm-adjusted p"] = holm(M["Raw p"])
    M["Decision at 0.05"] = np.where(M["Holm-adjusted p"] < 0.05, "Different", "Not detected")
    M.to_csv(os.path.join(config.TABLES, "T_K_external_mcnemar.csv"), index=False); print(M.to_string(index=False))
