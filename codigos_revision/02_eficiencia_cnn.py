# -*- coding: utf-8 -*-
"""
02_eficiencia_cnn.py — Eficiencia computacional de los clasificadores (Tabla 7 y Tabla 11)
Versión revisada de CNN.py. Mantiene EXACTAMENTE el protocolo publicado:
  * GPU NVIDIA RTX 3060, backend Keras = PyTorch (CUDA)
  * 562 imágenes del test precargadas como tensores en VRAM (sin E/S de disco)
  * batch = 1, 30 inferencias de calentamiento no cronometradas
  * torch.cuda.synchronize() antes y después de cada inferencia
  * solo forward pass: sin lectura, preprocesamiento ni posprocesamiento
AÑADIDO respecto a CNN.py (lo que faltaba para el artículo y para R1-2 / R2-10):
  * número de parámetros, tamaño FP32 (parámetros x 4 bytes) y tamaño del archivo nativo
  * imágenes/s = 1000 / latencia media (Ecuación 12)
  * soporte para los baselines nuevos (Keras y PyTorch/timm), con el mismo protocolo
  * registro de versiones, nvidia-smi y checksum SHA-256 de cada modelo
Salidas (resultados_revision/):
  tablas/tabla_resumen_cnn.csv            -> mismo formato que la tabla original + columnas nuevas
  tablas/tiempos_individuales_cnn.csv     -> una columna por modelo, una fila por imagen
  entorno/versiones_cnn.json, entorno/nvidia_smi.txt
"""
import os
import config                                   # fija KERAS_BACKEND=torch antes de importar keras
import sys, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import keras

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# 1. Verificación obligatoria de hardware (igual que CNN.py)
# ---------------------------------------------------------------------------
print("=" * 75)
print(" BENCHMARK DE INFERENCIA CNN (Keras con backend PyTorch / timm)")
print("=" * 75)
device, gpu_name = config.require_gpu()
env = config.save_environment("cnn")
print(f"[-] Keras {env['keras']} | PyTorch {env['torch']} | CUDA {env.get('cuda')} | GPU {gpu_name} ({env.get('vram_gb')} GB)")
print(f"[-] Backend de Keras: {keras.backend.backend()}")
if keras.backend.backend() != "torch":
    raise SystemError("Keras debe usar el backend 'torch' para que las latencias sean comparables con las publicadas.")

# ---------------------------------------------------------------------------
# 2. Imágenes del test (562) -> tensores en VRAM
#    Se preparan dos versiones: NHWC 0-255 (Keras) y NCHW normalizada (timm).
#    El preprocesamiento se hace AQUÍ, fuera del cronómetro.
# ---------------------------------------------------------------------------
image_paths = config.list_images(config.CLS_TEST_DIR)
print(f"[-] Imágenes en {config.CLS_TEST_DIR}: {len(image_paths)} (esperadas: 562)")
if not image_paths:
    raise FileNotFoundError(config.CLS_TEST_DIR)
arrays = [np.asarray(Image.open(p).convert("RGB").resize(config.CNN_SIZE), dtype=np.float32) for p in image_paths]

def tensors_for(cfg, timm_model=None):
    """Tensores (1, ...) en GPU según el preprocesamiento del modelo."""
    if cfg["framework"] == "timm":
        import timm
        dc = timm.data.resolve_data_config({}, model=timm_model)
        mean = np.array(dc["mean"], np.float32); std = np.array(dc["std"], np.float32)
        return [torch.from_numpy(((a / 255.0 - mean) / std).transpose(2, 0, 1)[None].copy()).to(device) for a in arrays]
    if cfg.get("preproc") == "tf":                       # ResNetV2 / Xception: [-1, 1]
        return [torch.from_numpy((a / 127.5 - 1.0)[None]).to(device) for a in arrays]
    return [torch.from_numpy(a[None]).to(device) for a in arrays]   # 0-255, igual que CNN.py

def load(cfg):
    """Carga el modelo y devuelve (callable, nº de parámetros, objeto)."""
    if cfg["framework"] == "keras":
        m = keras.models.load_model(cfg["path"], compile=False)
        return (lambda x: m(x, training=False)), m.count_params(), m
    import timm
    m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=4)
    m.load_state_dict(torch.load(cfg["path"], map_location="cpu")); m.eval().to(device)
    return m, sum(p.numel() for p in m.parameters()), m

# ---------------------------------------------------------------------------
# 3. Medición
# ---------------------------------------------------------------------------
raw, rows = {}, []
with torch.no_grad():
    for name, cfg in config.available(config.CNN_MODELS).items():
        print(f"\n>>> {name}\n    Ruta: {cfg['path']}")
        fwd, n_params, obj = load(cfg)
        X = tensors_for(cfg, obj if cfg["framework"] == "timm" else None)

        for _ in range(config.N_WARMUP):              # calentamiento (no se registra)
            fwd(X[0]); torch.cuda.synchronize()

        lat = []
        for x in X:                                   # una imagen por inferencia (batch = 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            fwd(x)
            torch.cuda.synchronize()                  # esperar a que la GPU termine
            lat.append((time.perf_counter_ns() - t0) / 1e6)   # ns -> ms
        raw[name] = lat

        mean = float(np.mean(lat))
        # Desviación estándar poblacional (np.std, ddof=0), igual que en los valores publicados
        std, med = float(np.std(lat)), float(np.median(lat))
        size_native = os.path.getsize(cfg["path"]) / 1e6
        rows.append({
            "Modelo": name,
            "Media (ms/img)": f"{mean:.4f}",
            "Desv. estándar (ms)": f"{std:.4f}",
            "Mediana (ms)": f"{med:.4f}",
            "N": len(lat), "Input": "224x224", "Batch": config.BATCH_SIZE, "Hardware": gpu_name,
            # ---- columnas nuevas para la Tabla 7 ----
            "Framework": "Keras (torch backend)" if cfg["framework"] == "keras" else "PyTorch/timm",
            "Parámetros": int(n_params),
            "Parámetros (M)": round(n_params / 1e6, 2),
            "Archivo nativo (MB)": round(size_native, 2),
            "FP32 solo pesos (MB)": round(n_params * 4 / 1e6, 2),     # parámetros x 4 bytes
            "Imágenes/s": round(1000.0 / mean, 2),                   # Ecuación (12)
            "SHA-256": config.sha256(cfg["path"]),
        })
        print(f"    [+] media {mean:.4f} ms | DE {std:.4f} | mediana {med:.4f} | {1000/mean:.2f} img/s | {n_params:,} parámetros")
        del obj; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 4. Exportación
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(os.path.join(config.TABLES, "tabla_resumen_cnn.csv"), index=False)
pd.DataFrame(raw).to_csv(os.path.join(config.TABLES, "tiempos_individuales_cnn.csv"), index=False)
print("\n" + df.drop(columns=["SHA-256", "Hardware"]).to_string(index=False))
print(f"\n[-] Protocolo: batch={config.BATCH_SIZE}, warm-up={config.N_WARMUP}, sincronización GPU activa, tensores precargados en VRAM")
print("[-] Control: ResNet50V2 ≈ 255.65 ms, Xception ≈ 185.62 ms, MicroNet_Banana ≈ 18.87 ms (valores publicados)")
