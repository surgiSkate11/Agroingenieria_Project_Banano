# -*- coding: utf-8 -*-
"""
09_entrenar_baselines_yolo.py — Detectores nuevos (R1-2): YOLOv5n, YOLOv9t y RT-DETR-L
NUEVO. Misma configuración que YOLOv8n/v10n/v11n (Sección 3.5): 640 px, 100 épocas, batch 16, SGD, lr0 = 0,01,
semilla fija. Excepción declarada: RT-DETR usa AdamW (lr0 = 1e-4) y batch 8 por la memoria de 12 GB.
Los pesos quedan en PROJECT_ROOT/models/YOLO/<CARPETA>/weights/best.pt (rutas de config.py).
Después ejecutar 03, 06 y 07.
Uso: python 09_entrenar_baselines_yolo.py --model yolov5n
"""
import os, argparse
import config
from ultralytics import YOLO, RTDETR

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["yolov5n", "yolov9t", "rtdetr-l"])
    A = ap.parse_args(); config.set_seed(); config.require_gpu()
    project = os.path.join(config.PROJECT_ROOT, "models", "YOLO")
    if A.model == "rtdetr-l":
        RTDETR("rtdetr-l.pt").train(data=config.DET_YAML, epochs=100, imgsz=640, batch=8, optimizer="AdamW", lr0=1e-4,
                                    seed=config.SEED, deterministic=True, project=project, name="RTDETR_L", exist_ok=True)
    else:
        weights = {"yolov5n": "yolov5nu.pt", "yolov9t": "yolov9t.pt"}[A.model]   # pesos preentrenados de Ultralytics
        YOLO(weights).train(data=config.DET_YAML, epochs=100, imgsz=640, batch=16, optimizer="SGD", lr0=0.01,
                            seed=config.SEED, deterministic=True, project=project, name=A.model.upper(), exist_ok=True)
    print("[+] Listo. Pesos en", project)

if __name__ == "__main__":   # obligatorio en Windows (los workers de Ultralytics reimportan el script)
    main()
