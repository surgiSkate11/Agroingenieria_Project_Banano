# -*- coding: utf-8 -*-
"""
06_predicciones_metricas_yolo.py — Detecciones POR IMAGEN y métricas por clase en el test (757 imágenes)
NUEVO. Usa los pesos YA ENTRENADOS (no reentrena). Faltaba en el manuscrito: sin las detecciones
por imagen no se puede hacer la comparación pareada de detectores (Sección 4.4, Tabla 9; R1-2, R1-4).
Para cada detector:
  * model.val(split="test", save_json=True) -> predictions.json (formato COCO)
  * precisión, recall, mAP@50, mAP@50-95 global y por clase; F1 = 2PR/(P+R) (Ecuación 3)
Salidas:
  predicciones/det_<Modelo>.json
  tablas/det_metricas_<Modelo>.csv y tablas/T_G0_detection_overall_ultralytics.csv (amplía la Tabla 6)
Requisito: data.yaml debe tener la línea  test: test/images  (además de train y val).
"""
import os, shutil
import config
import pandas as pd
from cargar_modelos import load_detector

config.require_gpu(); config.save_environment("yolo_val")
overall = []
for name, cfg in config.available(config.YOLO_MODELS).items():
    print(f"\n>>> {name}")
    m = load_detector(name)
    # conf=0.001 e iou=0.7: valores por defecto de validación de Ultralytics (necesarios para el mAP)
    r = m.val(data=config.DET_YAML, split="test", imgsz=640, batch=1, conf=0.001, iou=0.7,
              save_json=True, plots=True, workers=0, project=os.path.join(config.OUT, "yolo_val"), name=name, exist_ok=True)
    shutil.copy(os.path.join(str(r.save_dir), "predictions.json"), os.path.join(config.PREDS, f"det_{name}.json"))
    P, R = 100 * r.box.mp, 100 * r.box.mr
    rows = [{"Model": name, "Class": "all", "Precision %": P, "Recall %": R,
             "mAP@50 %": 100 * r.box.map50, "mAP@50–95 %": 100 * r.box.map}]
    for i, c in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap = r.box.class_result(i)
        rows.append({"Model": name, "Class": m.names[int(c)], "Precision %": 100 * p, "Recall %": 100 * rc,
                     "mAP@50 %": 100 * ap50, "mAP@50–95 %": 100 * ap})
    df = pd.DataFrame(rows)
    df["F1 %"] = 2 * df["Precision %"] * df["Recall %"] / (df["Precision %"] + df["Recall %"])
    df = df.round(2); df.to_csv(os.path.join(config.TABLES, f"det_metricas_{name}.csv"), index=False)
    overall.append(df.iloc[0]); print(df.to_string(index=False))
    # La matriz de confusión de Ultralytics (con fila "background") queda en yolo_val/<Modelo>/
T = pd.DataFrame(overall).drop(columns=["Class"])
T.to_csv(os.path.join(config.TABLES, "T_G0_detection_overall_ultralytics.csv"), index=False)
print("\n" + T.to_string(index=False))
print("Control (publicado): mAP@50 = 93.58 / 93.87 / 94.00 para YOLOv8n / YOLOv10n / YOLOv11n")
