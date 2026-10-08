# -*- coding: utf-8 -*-
"""
04_predicciones_cnn.py — Predicciones POR IMAGEN de los clasificadores en el test (562 imágenes)
NUEVO. Es la base de McNemar, TOST, matrices de confusión y análisis de Overripe (R1-2, R1-4).
Usa los modelos YA ENTRENADOS del manuscrito (no reentrena) y, si existen, los baselines nuevos.
Además resuelve la inconsistencia 536 vs. 537 aciertos de MicroNet_Banana.
Salidas:
  predicciones/cls_<Modelo>.csv : image_id, true_label, pred_label, prob_Unripe, prob_Ripe,
                                  prob_Overripe, prob_Rotten, correct
  tablas/control_aciertos_cnn.csv : aciertos obtenidos vs. publicados
"""
import os
import config
import numpy as np
import pandas as pd
from PIL import Image
import torch
import keras
from estadistica import paired_counts, mcnemar_exact

device, gpu = config.require_gpu()
paths = config.list_images(config.CLS_TEST_DIR)
# Etiqueta real = nombre de la carpeta (overripe, ripe, rotten, unripe)
true = [config.nice(os.path.basename(os.path.dirname(p))) for p in paths]
ids = [os.path.relpath(p, config.CLS_TEST_DIR).replace("\\", "/") for p in paths]
arrays = np.stack([np.asarray(Image.open(p).convert("RGB").resize(config.CNN_SIZE), dtype=np.float32) for p in paths])
print(f"[-] {len(paths)} imágenes de prueba; por clase:", pd.Series(true).value_counts().to_dict())

def predict(cfg, preproc):
    """Probabilidades (n, 4) en el orden MODEL_CLASS_ORDER."""
    if cfg["framework"] == "keras":
        m = keras.models.load_model(cfg["path"], compile=False)
        x = arrays / 127.5 - 1.0 if preproc == "tf" else arrays
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                y = m(torch.from_numpy(x[i:i + 32]).to(device), training=False)
                out.append(keras.ops.convert_to_numpy(y))
        return np.concatenate(out)
    import timm
    m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=4)
    m.load_state_dict(torch.load(cfg["path"], map_location="cpu")); m.eval().to(device)
    dc = timm.data.resolve_data_config({}, model=m)
    x = ((arrays / 255.0 - np.array(dc["mean"])) / np.array(dc["std"])).transpose(0, 3, 1, 2).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            out.append(torch.softmax(m(torch.from_numpy(x[i:i + 32]).to(device)), 1).cpu().numpy())
    return np.concatenate(out)

control = []
for name, cfg in config.available(config.CNN_MODELS).items():
    order = [config.nice(c) for c in config.MODEL_CLASS_ORDER]
    if cfg["framework"] == "timm":
        order = sorted(config.CLASSES)               # ImageFolder/timm: orden alfabético
    P = predict(cfg, cfg.get("preproc"))
    pred = [order[i] for i in P.argmax(1)]
    acc = np.mean(np.array(pred) == np.array(true))
    # Control automático: si la exactitud es muy baja, probablemente falta el preprocesamiento [-1, 1]
    if acc < 0.5 and cfg["framework"] == "keras" and cfg.get("preproc") is None:
        print(f"[aviso] {name}: exactitud {acc:.3f} con píxeles 0-255; se prueba preprocesamiento [-1, 1]")
        P2 = predict(cfg, "tf"); pred2 = [order[i] for i in P2.argmax(1)]
        if np.mean(np.array(pred2) == np.array(true)) > acc:
            P, pred = P2, pred2
            print(f"         -> usar preproc='tf' para {name} en config.py")
    df = pd.DataFrame({"image_id": ids, "true_label": true, "pred_label": pred})
    for c in config.CLASSES:
        df[f"prob_{c}"] = P[:, order.index(c)]
    df["correct"] = (df.true_label == df.pred_label).astype(int)
    df.to_csv(os.path.join(config.PREDS, f"cls_{name}.csv"), index=False)
    k = int(df.correct.sum()); exp = cfg.get("expected_correct")
    control.append({"Modelo": name, "Aciertos": k, "n": len(df), "Exactitud %": round(100 * k / len(df), 2),
                    "Publicado": exp if exp else "536 (verificar)" if name == "MicroNet_Banana" else "—",
                    "Coincide": ("Sí" if k == exp else "NO") if exp else "—"})
    print(f"[+] {name}: {k}/{len(df)} aciertos ({100*k/len(df):.2f} %)")

ctl = pd.DataFrame(control); ctl.to_csv(os.path.join(config.TABLES, "control_aciertos_cnn.csv"), index=False)
print("\n" + ctl.to_string(index=False))

# ---- Verificación 536 vs. 537 (Paso 0 de la guía) ----
L = lambda n: pd.read_csv(os.path.join(config.PREDS, f"cls_{n}.csv")).set_index("image_id")
mic, res, xce = L("MicroNet_Banana"), L("ResNet50V2"), L("Xception")
print("\nVERIFICACIÓN MicroNet_Banana (en el manuscrito: 536 aciertos; McNemar 2 vs 17 y 4 vs 20)")
for nm, o in [("ResNet50V2", res), ("Xception", xce)]:
    b, c = paired_counts(mic.correct.loc[o.index], o.correct)
    print(f"  MicroNet vs {nm}: solo MicroNet = {b}, solo {nm} = {c}, p exacto = {mcnemar_exact(b, c):.6f}; "
          f"aciertos {nm} - MicroNet = {int(o.correct.sum() - mic.correct.sum())} (debe ser igual a {c - b})")
print(f"  => Aciertos reales de MicroNet_Banana: {int(mic.correct.sum())}. Comunicar a Jhonny.")
