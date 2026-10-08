# -*- coding: utf-8 -*-
"""
12_gradcam_micronet.py — Grad-CAM de MicroNet_Banana en imágenes Overripe (OPCIONAL; R1-4)
NUEVO. Muestra si el modelo mira la cáscara (manchas, color) o el fondo en 4 errores y 2 aciertos.
Implementado para Keras con backend PyTorch (gradientes con torch.autograd).
Requiere predicciones/cls_MicroNet_Banana.csv (script 04). Salida: figuras/gradcam_micronet_overripe.png
"""
import os
import config
import numpy as np, pandas as pd, torch, keras
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

dev, _ = config.require_gpu()
m = keras.models.load_model(config.CNN_MODELS["MicroNet_Banana"]["path"], compile=False)
last_conv = [l for l in m.layers if isinstance(l, keras.layers.Conv2D)][-1]   # última convolución (bloque 4)
# Tras la convolución vienen BN y ReLU; se usa la salida de la convolución como en Grad-CAM estándar
grad_model = keras.Model(m.inputs, [last_conv.output, m.output])
d = pd.read_csv(os.path.join(config.PREDS, "cls_MicroNet_Banana.csv"))
o = d[d.true_label == "Overripe"]
sel = pd.concat([o[o.correct == 0].head(4), o[o.correct == 1].head(2)])
fig, axs = plt.subplots(2, len(sel), figsize=(2.2 * len(sel), 4.8))
for j, r in enumerate(sel.itertuples()):
    img = np.asarray(Image.open(os.path.join(config.CLS_TEST_DIR, r.image_id)).convert("RGB").resize(config.CNN_SIZE), np.float32)
    x = torch.from_numpy(img[None]).to(dev)
    with torch.enable_grad():
        conv, pred = grad_model(x, training=False)
        cls = int(torch.argmax(pred[0]))
        g = torch.autograd.grad(pred[0, cls], conv)[0][0]           # gradiente de la clase predicha
    w = g.mean(dim=(0, 1))                                          # pesos por canal (promedio espacial)
    cam = torch.relu((conv[0] * w).sum(-1)).detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-9)
    cam = np.asarray(Image.fromarray(np.uint8(255 * cam)).resize(config.CNN_SIZE)) / 255.0
    axs[0, j].imshow(img.astype("uint8")); axs[1, j].imshow(img.astype("uint8")); axs[1, j].imshow(cam, cmap="jet", alpha=0.45)
    prob = max(getattr(r, f"prob_{c}") for c in config.CLASSES)
    axs[0, j].set_title(f"True: Overripe\nPred: {r.pred_label} ({prob:.2f})", fontsize=8)
    for a in axs[:, j]:
        a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(config.FIGS, "gradcam_micronet_overripe.png"), dpi=600)
print("[+] figuras/gradcam_micronet_overripe.png")
