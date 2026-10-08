# -*- coding: utf-8 -*-
"""
13_tabla_eficiencia_articulo.py — Une las tablas de eficiencia (02 y 03) en el formato de la Tabla 7
NUEVO. Salida: tablas/T_H_efficiency.csv (encabezados en inglés, listos para el artículo).
"""
import os
import config
import pandas as pd

out = []
for task, f in [("Classification", "tabla_resumen_cnn.csv"), ("Detection", "tabla_resumen_yolo.csv")]:
    d = pd.read_csv(os.path.join(config.TABLES, f))
    for _, r in d.iterrows():
        out.append({"Task": task, "Model": r["Modelo"], "Framework": r["Framework"],
                    "Parameters (M)": r["Parámetros (M)"], "Native artifact (MB)": r["Archivo nativo (MB)"],
                    "Approx. FP32 weights (MB)": r["FP32 solo pesos (MB)"],
                    "Latency, mean ± SD (ms)": f"{float(r['Media (ms/img)']):.2f} ± {float(r['Desv. estándar (ms)']):.2f}",
                    "Median (ms)": round(float(r["Mediana (ms)"]), 2), "Images/s": r["Imágenes/s"], "N": r["N"]})
T = pd.DataFrame(out); T.to_csv(os.path.join(config.TABLES, "T_H_efficiency.csv"), index=False)
print(T.to_string(index=False))
