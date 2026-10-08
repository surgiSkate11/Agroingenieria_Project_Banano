# -*- coding: utf-8 -*-
"""
05_metricas_estadistica_cnn.py — Métricas, IC de Wilson, McNemar exacto + Holm, TOST y errores
NUEVO. Lee predicciones/cls_*.csv (script 04) y genera las tablas del artículo:
  T_A_classification_overall.csv  -> amplía la Tabla 5 (Ecuaciones 1-6)
  T_B_classwise.csv               -> Tabla S6 (sensibilidad, especificidad, BA por clase)
  T_C_overripe_errors.csv         -> a qué clases van los errores de Overripe (R1-4, Discusión)
  T_D_mcnemar_holm.csv            -> amplía la Tabla 8 (Ecuaciones 7-8)
  T_E_tost.csv                    -> equivalencia práctica (R1-4)
  T_F_micronet_coverage.csv       -> exactitud vs. cobertura de MicroNet (triaje con revisión humana)
  figuras/confusion_<Modelo>.png  -> matrices de confusión a 600 dpi
"""
import glob, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
import config
from config import CLASSES
from estadistica import wilson, mcnemar_exact, holm, paired_counts, tost_paired
PREDS, RESULTS, FIGS = config.PREDS, config.TABLES, config.FIGS

NAMES = {}   # los archivos ya usan el nombre del modelo (cls_<Modelo>.csv)
# Carga de todas las predicciones por imagen, alineadas por image_id
P = {os.path.basename(f)[4:-4]: pd.read_csv(f).set_index("image_id").sort_index()
     for f in sorted(glob.glob(os.path.join(PREDS, "cls_*.csv")))}
ref = next(iter(P.values())).index
for k, d in P.items():
    assert (d.index == ref).all(), f"{k}: las imágenes no coinciden"
n = len(ref)

# ---- T_A: desempeño global (amplía Tabla 5) ----
rows = []
for k, d in P.items():
    acc, lo, hi = wilson(int(d.correct.sum()), n)
    pr, rc, f1, _ = precision_recall_fscore_support(d.true_label, d.pred_label, labels=CLASSES,
                                                    average="macro", zero_division=0)
    rows.append({"Model": NAMES.get(k, k), "key": k, "Correct / n": f"{int(d.correct.sum())} / {n}",
                 "Accuracy % (95% Wilson CI)": f"{acc:.2f} ({lo:.2f}–{hi:.2f})",
                 "Macro precision %": round(100 * pr, 2), "Macro recall %": round(100 * rc, 2),
                 "Macro F1 %": round(100 * f1, 2), "_acc": acc})
TA = pd.DataFrame(rows).sort_values("_acc", ascending=False)
TA.drop(columns=["_acc", "key"]).to_csv(f"{RESULTS}/T_A_classification_overall.csv", index=False)
best = TA.key.iloc[0]; print(TA.drop(columns=["_acc"]))

# ---- T_B: métricas por clase (Tabla S6) ----
rows = []
for k, d in P.items():
    cm = confusion_matrix(d.true_label, d.pred_label, labels=CLASSES)
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        sen, spe = tp / (tp + fn), tn / (tn + fp); pre = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * pre * sen / (pre + sen) if pre + sen else 0
        rows.append({"Model": NAMES.get(k, k), "Class": c, "n": int(cm[i].sum()),
                     "Precision %": round(100 * pre, 2), "Sensitivity %": round(100 * sen, 2),
                     "Specificity %": round(100 * spe, 2), "Balanced accuracy %": round(50 * (sen + spe), 2),
                     "F1 %": round(100 * f1, 2)})
    # Matriz de confusión (CSV + figura)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]) \
      .to_csv(f"{RESULTS}/confusion_{k}.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), CLASSES, rotation=30); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(NAMES.get(k, k))
    fig.tight_layout(); fig.savefig(f"{FIGS}/confusion_{k}.png", dpi=600); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{RESULTS}/T_B_classwise.csv", index=False)

# ---- T_C: errores de Overripe (respuesta a R1-4) ----
rows = []
for k, d in P.items():
    o = d[d.true_label == "Overripe"]
    r = {"Model": NAMES.get(k, k), "Overripe n": len(o), "Correct": int((o.pred_label == "Overripe").sum())}
    for c in CLASSES:
        if c != "Overripe":
            r[f"→ {c}"] = int((o.pred_label == c).sum())
    r["Predicted Overripe but true ≠ Overripe"] = int(((d.pred_label == "Overripe") & (d.true_label != "Overripe")).sum())
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{RESULTS}/T_C_overripe_errors.csv", index=False)

# ---- T_D: McNemar exacto + Holm. Familia declarada: los 3 pares originales +
#      cada modelo nuevo vs. el mejor y vs. MicroNet_Banana ----
orig = [m for m in ("MicroNet_Banana", "ResNet50V2", "Xception") if m in P]
# Orden de cada par = orden del manuscrito (diferencia A - B): Xception - ResNet50V2 = +0.18 pp en el TOST
pairs = [p for p in [("MicroNet_Banana", "ResNet50V2"), ("MicroNet_Banana", "Xception"), ("Xception", "ResNet50V2")]
         if p[0] in P and p[1] in P]
for k in P:
    if k not in orig:
        pairs += [(k, best)] if k != best else []
        pairs += [(k, "MicroNet_Banana")] if "MicroNet_Banana" in P else []
pairs = list(dict.fromkeys(tuple(p) for p in pairs if p[0] != p[1]))
rows = []
for a, b in pairs:
    x, y = paired_counts(P[a].correct, P[b].correct)
    rows.append({"Comparison": f"{NAMES.get(a, a)} vs. {NAMES.get(b, b)}", "a": a, "b": b,
                 "Discordant counts": f"{x} vs. {y}", "Raw p": mcnemar_exact(x, y), "_b": x, "_c": y})
TD = pd.DataFrame(rows); TD["Holm-adjusted p"] = holm(TD["Raw p"])
TD["Decision at 0.05"] = np.where(TD["Holm-adjusted p"] < 0.05, "Different", "Not detected")
TD.drop(columns=["a", "b", "_b", "_c"]).to_csv(f"{RESULTS}/T_D_mcnemar_holm.csv", index=False)
# Familia ORIGINAL de 3 comparaciones (reproduce los valores ya publicados en la Tabla 8: Holm 0.002186 / 0.003088 / 1)
O3 = TD[TD.apply(lambda r: r["a"] in orig and r["b"] in orig, axis=1)].copy()
O3["Holm-adjusted p"] = holm(O3["Raw p"])
O3["Decision at 0.05"] = np.where(O3["Holm-adjusted p"] < 0.05, "Different", "Not detected")
O3.drop(columns=["a", "b", "_b", "_c"]).to_csv(f"{RESULTS}/T_D0_mcnemar_holm_original_family.csv", index=False)
print("Familia original (3 comparaciones):"); print(O3.drop(columns=["a", "b", "_b", "_c"]))
print(TD.drop(columns=["a", "b", "_b", "_c"]))
# Matriz completa (material suplementario)
keys = list(P); full = []
for a, b in itertools.combinations(keys, 2):
    x, y = paired_counts(P[a].correct, P[b].correct)
    full.append({"A": NAMES.get(a, a), "B": NAMES.get(b, b), "only A": x, "only B": y, "Raw p": mcnemar_exact(x, y)})
F = pd.DataFrame(full); F["Holm-adjusted p (all pairs)"] = holm(F["Raw p"])
F.to_csv(f"{RESULTS}/S_mcnemar_all_pairs.csv", index=False)

# ---- T_E: TOST (margen ±2 pp y ±1 pp) para pares sin diferencia detectada ----
rows = []
for _, r in TD.iterrows():
    if r["Holm-adjusted p"] >= 0.05:
        for m in (2.0, 1.0):
            t = tost_paired(r["_b"], r["_c"], n, m)
            rows.append({"Comparison": r["Comparison"], "Margin (pp)": f"±{m:g}",
                         "Difference (pp)": round(t["diff_pp"], 2),
                         "90% CI (pp)": f"{t['ci90_lo']:.2f} to {t['ci90_hi']:.2f}",
                         "TOST p": f"{t['p_tost']:.4f}" if t["p_tost"] >= 1e-3 else "< 0.001",
                         "Equivalent": "Yes" if t["equivalent"] else "No"})
pd.DataFrame(rows).to_csv(f"{RESULTS}/T_E_tost.csv", index=False); print(pd.DataFrame(rows))

# ---- T_F (opcional): cobertura vs. exactitud de MicroNet_Banana (triaje con revisión humana) ----
if "MicroNet_Banana" in P:
    d = P["MicroNet_Banana"]; conf = d[[f"prob_{c}" for c in CLASSES]].max(axis=1)
    rows = [{"Threshold τ": t, "Coverage %": round(100 * (conf >= t).mean(), 2),
             "Accuracy on covered %": round(100 * d.correct[conf >= t].mean(), 2),
             "Sent to manual review": int((conf < t).sum())}
            for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
    pd.DataFrame(rows).to_csv(f"{RESULTS}/T_F_micronet_coverage.csv", index=False)
