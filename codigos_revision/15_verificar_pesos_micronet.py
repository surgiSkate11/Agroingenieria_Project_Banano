# -*- coding: utf-8 -*-
"""
15_verificar_pesos_micronet.py — ¿Qué archivo de pesos de MicroNet_Banana produjo los resultados publicados?
NUEVO. El script 04 obtuvo 532 aciertos con el archivo que config.py elige primero
(Keras/best_MicroNet_Banana.keras); el manuscrito original reportaba 536 y McNemar 2 vs 17 y 4 vs 20.
Este script evalúa TODOS los archivos .keras de MicroNet que encuentre y reporta, para cada uno:
fecha de modificación, SHA-256, número de parámetros, aciertos en las 562 imágenes y conteos discordantes
frente a ResNet50V2 y Xception (usando predicciones/cls_ResNet50V2.csv y cls_Xception.csv del script 04).
NO modifica config.py ni las predicciones. Salida: tablas/control_pesos_micronet.csv
Uso: python 15_verificar_pesos_micronet.py
"""
import os, glob, datetime
import config
import numpy as np, pandas as pd
from PIL import Image
import torch, keras
from estadistica import paired_counts, mcnemar_exact

dev, _ = config.require_gpu()
cands = sorted({p for pat in ("**/*icro*et*.keras", "**/*MicroNet*.keras", "**/*micronet*.keras")
                for p in glob.glob(os.path.join(config.BASE_DIR, pat), recursive=True)})
print("Archivos candidatos:", *cands, sep="\n  ")
paths = config.list_images(config.CLS_TEST_DIR)
ids = [os.path.relpath(p, config.CLS_TEST_DIR).replace("\\", "/") for p in paths]
true = np.array([config.nice(os.path.basename(os.path.dirname(p))) for p in paths])
X = np.stack([np.asarray(Image.open(p).convert("RGB").resize(config.CNN_SIZE), np.float32) for p in paths])
order = [config.nice(c) for c in config.MODEL_CLASS_ORDER]
ref = {n: pd.read_csv(os.path.join(config.PREDS, f"cls_{n}.csv")).set_index("image_id").loc[ids].correct.values
       for n in ("ResNet50V2", "Xception")}
rows = []
for f in cands:
    m = keras.models.load_model(f, compile=False); out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            out.append(keras.ops.convert_to_numpy(m(torch.from_numpy(X[i:i + 32]).to(dev), training=False)))
    pred = np.array(order)[np.concatenate(out).argmax(1)]; ok = (pred == true).astype(int)
    r = {"Archivo": os.path.relpath(f, config.BASE_DIR),
         "Modificado": datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M"),
         "SHA-256": config.sha256(f)[:16], "Parámetros": m.count_params(), "Aciertos": int(ok.sum()),
         "Exactitud %": round(100 * ok.mean(), 2)}
    for n, o in ref.items():
        b, c = paired_counts(ok, o); r[f"vs {n} (solo MicroNet vs solo {n})"] = f"{b} vs {c} (p={mcnemar_exact(b, c):.6f})"
    rows.append(r); print(r)
pd.DataFrame(rows).to_csv(os.path.join(config.TABLES, "control_pesos_micronet.csv"), index=False)
print("\nSi algún archivo da 536 aciertos, avisar a Jhonny ANTES de cambiar config.py.")
