# -*- coding: utf-8 -*-
"""
config.py — Configuración ÚNICA para todos los scripts de la revisión
Manuscrito agriengineering-4601275 (MDPI AgriEngineering)

>>> EDITAR SOLO ESTE ARCHIVO si cambian rutas o modelos. <<<

Basado en los scripts originales CNN.py y yolo.py (mismas rutas, mismo protocolo).
Backend de Keras: PyTorch (igual que en CNN.py), para que TODAS las latencias se midan
con el mismo motor (PyTorch + CUDA) en la RTX 3060.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")   # DEBE ir antes de "import keras"
os.environ["PYTHONUNBUFFERED"] = "1"

import sys, json, platform, subprocess, random
import numpy as np
import pandas as _pd

# Las tablas T_* y S_* se guardan con BOM UTF-8 para que Excel muestre bien "±", "–" y "→".
# (Los CSV que otros scripts vuelven a leer -predicciones, control, resúmenes- se guardan sin BOM.)
_orig_to_csv = _pd.DataFrame.to_csv
def _to_csv(self, path_or_buf=None, *a, **k):
    if isinstance(path_or_buf, str) and os.path.basename(path_or_buf).startswith(("T_", "S_")):
        k.setdefault("encoding", "utf-8-sig")
    return _orig_to_csv(self, path_or_buf, *a, **k)
_pd.DataFrame.to_csv = _to_csv

# ---------------------------------------------------------------------------
# 1. RUTAS (las mismas de CNN.py y yolo.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = r"D:\Trabajo\Jhonny\articulo"            # raíz usada en yolo.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # carpeta de estos scripts (como CNN.py)

# Clasificación (Kaggle): carpeta con train/ valid/ test/ y subcarpetas por clase
CLS_DIR = os.path.join(BASE_DIR, "bananaDatasetNoYolo")
CLS_TEST_DIR = os.path.join(CLS_DIR, "test")             # 562 imágenes: overripe 113, ripe 154, rotten 185, unripe 110

# Detección (Roboflow v2, formato YOLO)
DET_DIR = os.path.join(PROJECT_ROOT, "bananos", "dataset1_yolo")
DET_TEST_IMAGES = os.path.join(DET_DIR, "test", "images")   # 757 imágenes
DET_TEST_LABELS = os.path.join(DET_DIR, "test", "labels")
DET_YAML = os.path.join(DET_DIR, "data.yaml")              # debe tener la clave "test:"

# Conjunto EXTERNO independiente: BananaImageBD (Ferdaus et al., Data in Brief 2025, 58, 111239; CC BY 4.0)
# Descargar de https://data.mendeley.com/datasets/ptfscwtnyz/2 y apuntar EXT_DIR a la carpeta que
# contiene las imágenes ORIGINALES (sin aumentar) del subconjunto de MADUREZ, con una subcarpeta por
# estado: Green, Semi-ripe, Ripe, Overripe  (los nombres se reconocen sin importar mayúsculas/guiones).
EXT_DIR = os.path.join(BASE_DIR, "BananaImageBD", "Ripeness", "Original")
# Mapeo de los estados de BananaImageBD a las clases del estudio.
# Semi-ripe no tiene equivalente (está entre Unripe y Ripe): se analiza por separado.
# BananaImageBD no tiene clase Rotten.
EXT_MAP = {"green": "Unripe", "unripe": "Unripe", "ripe": "Ripe", "overripe": "Overripe", "semiripe": None}

# Carpetas de salida (se crean solas)
OUT = os.path.join(BASE_DIR, "resultados_revision")
PREDS = os.path.join(OUT, "predicciones")      # predicciones por imagen (CLAVE para pruebas pareadas)
TABLES = os.path.join(OUT, "tablas")           # tablas CSV para el artículo
FIGS = os.path.join(OUT, "figuras")            # figuras a 600 dpi
LOGS = os.path.join(OUT, "entorno")            # versiones, nvidia-smi, checksums
for _d in (OUT, PREDS, TABLES, FIGS, LOGS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. PROTOCOLO COMÚN (Sección 3.7 del manuscrito)
# ---------------------------------------------------------------------------
SEED = 20260904
BATCH_SIZE = 1          # latencia con batch 1
N_WARMUP = 30           # inferencias de calentamiento no cronometradas
CNN_SIZE = (224, 224)
YOLO_SIZE = (640, 640)
EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# ---------------------------------------------------------------------------
# 3. CLASES
# ---------------------------------------------------------------------------
# Orden de SALIDA de los modelos Keras: image_dataset_from_directory ordena las carpetas
# alfabéticamente -> overripe, ripe, rotten, unripe.  VERIFICAR con el cuaderno de
# entrenamiento; si el orden fue otro, cambiarlo aquí (el script 04 avisa si la
# exactitud no coincide con la publicada).
MODEL_CLASS_ORDER = ["overripe", "ripe", "rotten", "unripe"]
# Orden de PRESENTACIÓN en las tablas del artículo
CLASSES = ["Unripe", "Ripe", "Overripe", "Rotten"]
def nice(c):   # "overripe" -> "Overripe"
    return c.strip().capitalize()

# Mapeo de las 6 clases de detección a las 4 de clasificación (validación cruzada)
DET_TO_CLS = {"freshunripe": "Unripe", "unripe": "Unripe",
              "freshripe": "Ripe", "ripe": "Ripe",
              "overripe": "Overripe", "rotten": "Rotten"}

def det_class(name):
    """Clase de 4 categorías a partir del nombre de la clase de detección (ignora mayúsculas, espacios y guiones)."""
    import re
    key = re.sub(r"[^a-z]", "", str(name).lower())
    if key not in DET_TO_CLS:
        raise KeyError(f"Clase de detección '{name}' sin mapeo: agregarla a DET_TO_CLS en config.py")
    return DET_TO_CLS[key]

def split_dir(split):
    """Carpeta de una partición del dataset de clasificación (acepta valid / val / validation)."""
    alias = {"valid": ("valid", "val", "validation"), "train": ("train",), "test": ("test",)}[split]
    for a in alias:
        p = os.path.join(CLS_DIR, a)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"No se encontró la partición '{split}' en {CLS_DIR} (se buscó: {alias})")

# ---------------------------------------------------------------------------
# 4. REGISTRO DE MODELOS
#    required=True  -> modelo del manuscrito (si falta el archivo, se detiene)
#    required=False -> baseline nuevo (si aún no existe el archivo, se omite con aviso)
#    preproc: preprocesamiento que se aplica ANTES de entrar al modelo
#             None   = el modelo recibe píxeles 0-255 (como en CNN.py; MicroNet tiene Rescaling interno)
#             "tf"   = escala a [-1, 1] (preprocess_input de ResNetV2 y Xception)
#             "timm" = normalización media/desv. del modelo timm
#    expected_correct: aciertos publicados en 562 imágenes (control de que todo está bien)
# ---------------------------------------------------------------------------
def _first(*cands):
    for c in cands:
        p = c if os.path.isabs(c) else os.path.join(BASE_DIR, c)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, cands[0])

CNN_MODELS = {
    # ----- modelos del manuscrito (rutas tomadas de CNN.py) -----
    "ResNet50V2": dict(path=_first(os.path.join("Keras", "resnet_model.keras"), os.path.join("modelos", "resnet50v2.keras"), "resnet_model.keras"),
                       framework="keras", preproc=None, required=True, expected_correct=552),
    "Xception": dict(path=_first(os.path.join("Keras", "xception_model.keras"), os.path.join("modelos", "xception.keras"), "xception_model.keras"),
                     framework="keras", preproc=None, required=True, expected_correct=553),
    "MicroNet_Banana": dict(path=_first(os.path.join("Keras", "best_MicroNet_Banana.keras"), os.path.join("Keras", "MicroNet_Banana.keras"),
                                        os.path.join("modelos", "micronet_banana.keras"), "MicroNet_Banana.keras"),
                            framework="keras", preproc=None, required=True, expected_correct=None),  # 536 o 537: SE VERIFICA
    # ----- baselines nuevos (R1-2); se generan con 08a/08b -----
    "MobileNetV3-Large": dict(path=os.path.join(BASE_DIR, "Keras", "mobilenetv3large.keras"), framework="keras", preproc=None, required=False),
    "MobileNetV3-Small": dict(path=os.path.join(BASE_DIR, "Keras", "mobilenetv3small.keras"), framework="keras", preproc=None, required=False),
    "EfficientNet-B0": dict(path=os.path.join(BASE_DIR, "Keras", "efficientnetb0.keras"), framework="keras", preproc=None, required=False),
    "DeiT-Tiny": dict(path=os.path.join(BASE_DIR, "PyTorch", "deit_tiny_patch16_224.pt"), framework="timm",
                      timm_name="deit_tiny_patch16_224", preproc="timm", required=False),
    "MobileViT-S": dict(path=os.path.join(BASE_DIR, "PyTorch", "mobilevit_s.pt"), framework="timm",
                        timm_name="mobilevit_s", preproc="timm", required=False),
}

YOLO_MODELS = {
    # ----- modelos del manuscrito (rutas tomadas de yolo.py) -----
    "YOLOv8n": dict(path=os.path.join(PROJECT_ROOT, "models", "YOLO", "YOLOV8", "weights", "yolo8 best.pt"), kind="yolo", required=True),
    "YOLOv10n": dict(path=os.path.join(PROJECT_ROOT, "models", "YOLO", "YOLOV10", "weights", "yolo 10 best.pt"), kind="yolo", required=True),
    "YOLOv11n": dict(path=os.path.join(PROJECT_ROOT, "models", "YOLO", "YOLOV11", "weights", "yolo11 best.pt"), kind="yolo", required=True),
    # ----- baselines nuevos (R1-2); se generan con 09_entrenar_baselines_yolo.py -----
    "YOLOv5n": dict(path=os.path.join(PROJECT_ROOT, "models", "YOLO", "YOLOV5N", "weights", "best.pt"), kind="yolo", required=False),
    "YOLOv9t": dict(path=os.path.join(PROJECT_ROOT, "models", "YOLO", "YOLOV9T", "weights", "best.pt"), kind="yolo", required=False),
    "RT-DETR-L": dict(path=os.path.join(PROJECT_ROOT, "models", "YOLO", "RTDETR_L", "weights", "best.pt"), kind="rtdetr", required=False),
}

def available(models):
    """Devuelve solo los modelos cuyo archivo existe; detiene si falta uno obligatorio."""
    out = {}
    for name, cfg in models.items():
        if os.path.exists(cfg["path"]):
            out[name] = cfg
        elif cfg.get("required"):
            raise FileNotFoundError(f"No se encontró el modelo obligatorio {name}: {cfg['path']}")
        else:
            print(f"[aviso] {name}: archivo no encontrado, se omite ({cfg['path']})", flush=True)
    return out

# ---------------------------------------------------------------------------
# 5. UTILIDADES
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    """Semilla fija en Python, NumPy, PyTorch y Keras (Sección 3.4)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except Exception:
        pass

def list_images(folder):
    """Lista ordenada de imágenes (recursiva), igual que en CNN.py / yolo.py."""
    import glob
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.*"), recursive=True))
    return [p for p in paths if p.lower().endswith(EXTS)]

def require_gpu():
    """Verificación obligatoria de GPU (igual que los scripts originales)."""
    import torch
    if not torch.cuda.is_available():
        raise SystemError("[ERROR CRÍTICO] No se detectó GPU NVIDIA con CUDA. Abortando.")
    torch.cuda.init(); torch.cuda.set_device(0)
    return torch.device("cuda:0"), torch.cuda.get_device_name(0)

def save_environment(tag):
    """Guarda versiones de librerías y la salida de nvidia-smi (evidencia pedida para el artículo)."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "keras", "tensorflow", "ultralytics", "timm", "numpy", "pandas", "scipy", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "no instalado"
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU"
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None
    except Exception:
        pass
    json.dump(info, open(os.path.join(LOGS, f"versiones_{tag}.json"), "w"), indent=2)
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        open(os.path.join(LOGS, "nvidia_smi.txt"), "w").write(smi)
    except Exception:
        pass
    return info

def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
