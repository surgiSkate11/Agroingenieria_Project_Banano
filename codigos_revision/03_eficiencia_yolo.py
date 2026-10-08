# -*- coding: utf-8 -*-
"""
03_eficiencia_yolo.py — Eficiencia computacional de los detectores (Tabla 7 y Tabla 11)
Versión revisada de yolo.py. Mismo protocolo publicado:
  * 757 imágenes del test, 640 x 640, normalizadas a [0, 1], NCHW, precargadas en VRAM
  * batch = 1, 30 inferencias de calentamiento, torch.cuda.synchronize() antes y después
  * solo la red (forward pass): SIN NMS ni otro posprocesamiento
AÑADIDO respecto a yolo.py:
  * parámetros, tamaño FP32 y tamaño del archivo .pt; imágenes/s (Ecuación 12)
  * baselines nuevos: YOLOv5n, YOLOv9t y RT-DETR-L (R1-2)
  * versiones, nvidia-smi y SHA-256 de los pesos
Salidas: tablas/tabla_resumen_yolo.csv, tablas/tiempos_individuales_yolo.csv, entorno/versiones_yolo.json
"""
import os
import config
import sys, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO, RTDETR

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("=" * 75)
print(" BENCHMARK DE INFERENCIA YOLO / RT-DETR (Ultralytics / PyTorch)")
print("=" * 75)
device, gpu_name = config.require_gpu()
env = config.save_environment("yolo")
print(f"[-] PyTorch {env['torch']} | Ultralytics {env['ultralytics']} | CUDA {env.get('cuda')} | GPU {gpu_name}")

# ---------------------------------------------------------------------------
# Imágenes del test (757): redimensionadas, [0, 1], NCHW, en VRAM (fuera del cronómetro)
# ---------------------------------------------------------------------------
image_paths = config.list_images(config.DET_TEST_IMAGES)
print(f"[-] Imágenes en {config.DET_TEST_IMAGES}: {len(image_paths)} (esperadas: 757)")
if not image_paths:
    raise FileNotFoundError(config.DET_TEST_IMAGES)
X = []
for p in image_paths:
    a = np.asarray(Image.open(p).convert("RGB").resize(config.YOLO_SIZE), dtype=np.float32) / 255.0
    X.append(torch.from_numpy(a.transpose(2, 0, 1)[None].copy()).to(device))

raw, rows = {}, []
with torch.no_grad():
    for name, cfg in config.available(config.YOLO_MODELS).items():
        print(f"\n>>> {name}\n    Ruta: {cfg['path']}")
        wrapper = RTDETR(cfg["path"]) if cfg["kind"] == "rtdetr" else YOLO(cfg["path"])
        net = wrapper.model.to(device).eval()        # red pura, sin el pipeline predict() (que incluye NMS)
        n_params = sum(p.numel() for p in net.parameters())

        for _ in range(config.N_WARMUP):
            net(X[0]); torch.cuda.synchronize()

        lat = []
        for x in X:
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            net(x)
            torch.cuda.synchronize()
            lat.append((time.perf_counter_ns() - t0) / 1e6)
        raw[name] = lat
        mean, std, med = float(np.mean(lat)), float(np.std(lat)), float(np.median(lat))   # DE poblacional, como lo publicado
        rows.append({
            "Modelo": name, "Media (ms/img)": f"{mean:.4f}", "Desv. estándar (ms)": f"{std:.4f}",
            "Mediana (ms)": f"{med:.4f}", "N": len(lat), "Input": "640x640", "Batch": config.BATCH_SIZE,
            "Hardware": gpu_name, "Framework": f"Ultralytics {env['ultralytics']}",
            "Parámetros": int(n_params), "Parámetros (M)": round(n_params / 1e6, 2),
            "Archivo nativo (MB)": round(os.path.getsize(cfg["path"]) / 1e6, 2),
            "FP32 solo pesos (MB)": round(n_params * 4 / 1e6, 2),
            "Imágenes/s": round(1000.0 / mean, 2),
            "SHA-256": config.sha256(cfg["path"]),
        })
        print(f"    [+] media {mean:.4f} ms | DE {std:.4f} | mediana {med:.4f} | {1000/mean:.2f} img/s | {n_params:,} parámetros")
        del wrapper, net; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(config.TABLES, "tabla_resumen_yolo.csv"), index=False)
pd.DataFrame(raw).to_csv(os.path.join(config.TABLES, "tiempos_individuales_yolo.csv"), index=False)
print("\n" + df.drop(columns=["SHA-256", "Hardware"]).to_string(index=False))
print("[-] Control: YOLOv8n ≈ 27.12 ms, YOLOv10n ≈ 40.81 ms, YOLOv11n ≈ 42.62 ms (valores publicados)")
print("[-] Nota: los parámetros pueden diferir ligeramente de la Tabla 7 si la red no está fusionada (Conv+BN).")
