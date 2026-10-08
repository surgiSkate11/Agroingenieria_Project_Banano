# -*- coding: utf-8 -*-
"""cargar_modelos.py — Carga uniforme de clasificadores y detectores para los scripts 10, 11 y 12."""
import config
import numpy as np
import torch

def load_classifier(name):
    """Devuelve f(lista de PIL.Image) -> probabilidades (n, 4) en el orden config.CLASSES."""
    cfg = config.CNN_MODELS[name]
    dev = torch.device("cuda:0")
    if cfg["framework"] == "keras":
        import keras
        m = keras.models.load_model(cfg["path"], compile=False)
        order = [config.nice(c) for c in config.MODEL_CLASS_ORDER]
        def f(imgs):
            x = np.stack([np.asarray(i.convert("RGB").resize(config.CNN_SIZE), np.float32) for i in imgs])
            if cfg.get("preproc") == "tf":
                x = x / 127.5 - 1.0
            with torch.no_grad():
                p = keras.ops.convert_to_numpy(m(torch.from_numpy(x).to(dev), training=False))
            return p[:, [order.index(c) for c in config.CLASSES]]
        return f
    import timm
    m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=4)
    m.load_state_dict(torch.load(cfg["path"], map_location="cpu")); m.eval().to(dev)
    dc = timm.data.resolve_data_config({}, model=m)
    mean, std = np.array(dc["mean"], np.float32), np.array(dc["std"], np.float32)
    order = sorted(config.CLASSES)                    # timm / ImageFolder: orden alfabético
    def f(imgs):
        x = np.stack([(np.asarray(i.convert("RGB").resize(config.CNN_SIZE), np.float32) / 255 - mean) / std for i in imgs])
        with torch.no_grad():
            p = torch.softmax(m(torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(dev)), 1).cpu().numpy()
        return p[:, [order.index(c) for c in config.CLASSES]]
    return f

def load_detector(name):
    from ultralytics import YOLO, RTDETR
    cfg = config.YOLO_MODELS[name]
    return RTDETR(cfg["path"]) if cfg["kind"] == "rtdetr" else YOLO(cfg["path"])

def top1_class(det_model, image_path):
    """Clase (4 clases) de la detección de mayor confianza; 'none' si no detecta nada."""
    r = det_model.predict(image_path, imgsz=640, conf=0.001, verbose=False)[0]
    if len(r.boxes) == 0:
        return "none"
    name = det_model.names[int(r.boxes.cls[r.boxes.conf.argmax()])]
    return config.det_class(name)
