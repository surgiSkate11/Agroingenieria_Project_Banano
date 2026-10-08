# -*- coding: utf-8 -*-
"""
08a_entrenar_baselines_cnn_keras.py — Baselines nuevos de clasificación en Keras (R1-2)
NUEVO. MobileNetV3-Large, MobileNetV3-Small y EfficientNet-B0 (CNN ligeras), con el MISMO protocolo
de ResNet50V2 y Xception (Sección 3.4, Tabla 4):
  * entrada 224 x 224, pesos ImageNet, batch 64, entropía cruzada categórica, semilla 20260904
  * fase 1: base congelada, Adam, hasta 20 épocas
  * fase 2: últimas 15 capas descongeladas (BatchNorm congeladas), Adam lr = 1e-5
  * early stopping sobre val_loss con restauración de los mejores pesos
Backend de Keras = torch (igual que en la medición de latencia) -> el modelo se guarda en Keras/<nombre>.keras,
que es la ruta registrada en config.py. Después ejecutar 02, 04 y 05.
Uso: python 08a_entrenar_baselines_cnn_keras.py --model mobilenetv3large
"""
import os, argparse, json, time
import config
import keras
from keras import layers

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["mobilenetv3large", "mobilenetv3small", "efficientnetb0"])
ap.add_argument("--lr_fase1", type=float, default=1e-3)   # usar la misma lr de fase 1 que los cuadernos originales
A = ap.parse_args(); config.set_seed(); config.require_gpu()
NAME = {"mobilenetv3large": "MobileNetV3-Large", "mobilenetv3small": "MobileNetV3-Small", "efficientnetb0": "EfficientNet-B0"}[A.model]
BUILD = {"mobilenetv3large": keras.applications.MobileNetV3Large,   # incluyen su propio preprocesamiento:
         "mobilenetv3small": keras.applications.MobileNetV3Small,   # reciben píxeles 0-255, como MicroNet
         "efficientnetb0": keras.applications.EfficientNetB0}[A.model]

def ds(split, shuffle):
    # Mismo lector que el entrenamiento original; class_names en el orden de config.MODEL_CLASS_ORDER
    return keras.utils.image_dataset_from_directory(config.split_dir(split), label_mode="categorical",
                                                    class_names=config.MODEL_CLASS_ORDER, image_size=config.CNN_SIZE,
                                                    batch_size=64, shuffle=shuffle, seed=config.SEED)
tr, va = ds("train", True), ds("valid", False)

base = BUILD(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg")
inp = keras.Input((224, 224, 3))
x = base(inp, training=False)
x = layers.Dropout(0.3)(x)                       # misma cabeza que ResNet50V2/Xception (ajustar si fue otra)
out = layers.Dense(4, activation="softmax")(x)
model = keras.Model(inp, out, name=NAME)
cbs = lambda pat: [keras.callbacks.EarlyStopping("val_loss", patience=pat, restore_best_weights=True),
                   keras.callbacks.CSVLogger(os.path.join(config.LOGS, f"train_{A.model}.csv"), append=True)]
t0 = time.time()
base.trainable = False                            # ---- fase 1
model.compile(keras.optimizers.Adam(A.lr_fase1), "categorical_crossentropy", ["accuracy"])
h1 = model.fit(tr, validation_data=va, epochs=20, callbacks=cbs(5))
base.trainable = True                             # ---- fase 2
for l in base.layers[:-15]:
    l.trainable = False
for l in base.layers:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False
model.compile(keras.optimizers.Adam(1e-5), "categorical_crossentropy", ["accuracy"])
h2 = model.fit(tr, validation_data=va, epochs=60, callbacks=cbs(10))

os.makedirs(os.path.join(config.BASE_DIR, "Keras"), exist_ok=True)
path = os.path.join(config.BASE_DIR, "Keras", f"{A.model}.keras"); model.save(path)
json.dump({"model": NAME, "params": int(model.count_params()), "epochs_phase1": len(h1.history["loss"]),
           "epochs_phase2": len(h2.history["loss"]), "lr_phase1": A.lr_fase1, "lr_phase2": 1e-5,
           "train_minutes": round((time.time() - t0) / 60, 1), "sha256": config.sha256(path)},
          open(os.path.join(config.LOGS, f"info_{A.model}.json"), "w"), indent=2)
print(f"[+] Guardado {path}")
