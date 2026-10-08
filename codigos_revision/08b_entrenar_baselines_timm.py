# -*- coding: utf-8 -*-
"""
08b_entrenar_baselines_timm.py — Baselines Transformer e híbrido en PyTorch/timm (R1-2)
NUEVO. DeiT-Tiny (Transformer) y MobileViT-S (híbrido CNN-Transformer), mismo protocolo:
  * 224 x 224, pesos ImageNet, batch 64, semilla 20260904, sin aumentos adicionales
    (el dataset ya trae los aumentos del proveedor)
  * fase 1: solo la cabeza, AdamW lr 1e-3, hasta 20 épocas (paciencia 5)
  * fase 2: todo el modelo, AdamW lr 1e-5, hasta 60 épocas (paciencia 10)
  * early stopping por pérdida de validación, restaurando el mejor estado
Guarda PyTorch/<timm_name>.pt (ruta registrada en config.py). Después ejecutar 02, 04 y 05.
Uso: python 08b_entrenar_baselines_timm.py --model deit_tiny_patch16_224
"""
import os, argparse, copy, json, time
import config
import torch, timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["deit_tiny_patch16_224", "mobilevit_s"])
A = ap.parse_args(); config.set_seed(); dev, _ = config.require_gpu()
model = timm.create_model(A.model, pretrained=True, num_classes=4).to(dev)
dc = timm.data.resolve_data_config({}, model=model)
tf = transforms.Compose([transforms.Resize(config.CNN_SIZE), transforms.ToTensor(), transforms.Normalize(dc["mean"], dc["std"])])
g = torch.Generator().manual_seed(config.SEED)
# ImageFolder ordena alfabéticamente: overripe, ripe, rotten, unripe (lo asume cargar_modelos.py)
tr = DataLoader(datasets.ImageFolder(config.split_dir("train"), tf), 64, shuffle=True, generator=g, num_workers=0)
va = DataLoader(datasets.ImageFolder(config.split_dir("valid"), tf), 64, num_workers=0)
print("Clases:", tr.dataset.classes)
lossf = torch.nn.CrossEntropyLoss()

def phase(params, lr, epochs, patience, frozen_backbone=False):
    opt = torch.optim.AdamW(params, lr=lr); best, bad, state, n = 1e9, 0, None, 0
    for ep in range(epochs):
        model.train()
        if frozen_backbone:
            model.eval()          # fase 1: base congelada también en BN (equivale a training=False de Keras)
        for x, y in tr:
            opt.zero_grad(); loss = lossf(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in va:
                vl += lossf(model(x.to(dev)), y.to(dev)).item() * len(y)
        vl /= len(va.dataset); n = ep + 1; print(f"  época {n}: val_loss = {vl:.4f}", flush=True)
        if vl < best:
            best, bad, state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); return n

t0 = time.time()
for p in model.parameters(): p.requires_grad = False          # fase 1: solo la cabeza
for p in model.get_classifier().parameters(): p.requires_grad = True
e1 = phase(model.get_classifier().parameters(), 1e-3, 20, 5, frozen_backbone=True)
for p in model.parameters(): p.requires_grad = True           # fase 2: todo el modelo
e2 = phase(model.parameters(), 1e-5, 60, 10)

os.makedirs(os.path.join(config.BASE_DIR, "PyTorch"), exist_ok=True)
path = os.path.join(config.BASE_DIR, "PyTorch", f"{A.model}.pt"); torch.save(model.state_dict(), path)
json.dump({"model": A.model, "params": sum(p.numel() for p in model.parameters()), "epochs_phase1": e1, "epochs_phase2": e2,
           "train_minutes": round((time.time() - t0) / 60, 1), "timm": timm.__version__, "sha256": config.sha256(path)},
          open(os.path.join(config.LOGS, f"info_{A.model}.json"), "w"), indent=2)
print(f"[+] Guardado {path}")
