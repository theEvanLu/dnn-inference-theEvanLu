"""
Train a simple DNN on Fashion-MNIST and export:
  • model/fashion_mnist.json  (architecture)
  • model/fashion_mnist.npz   (weights)
Run:
    python train_model.py
"""

import sys, pathlib, json, numpy as np, tensorflow as tf

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_FOLDER = BASE_DIR / "model"
MODEL_FOLDER.mkdir(exist_ok=True)

MODEL_BASENAME = "fashion_mnist"
JSON_PATH = MODEL_FOLDER / f"{MODEL_BASENAME}.json"
NPZ_PATH = MODEL_FOLDER / f"{MODEL_BASENAME}.npz"

EPOCHS, BATCH_SIZE, HIDDEN1, HIDDEN2 = 20, 128, 256, 128

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=40, batch_size=128, validation_split=0.1, callbacks=[callback])

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy = {test_acc:.4f}")

arch_list = []
weights_dict = {}

for lyr in model.layers:
    if isinstance(lyr, tf.keras.layers.Flatten):
        arch_list.append({
            "name": lyr.name,
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif isinstance(lyr, tf.keras.layers.Dense):
        W, b = lyr.get_weights()
        W_key, b_key = f"{lyr.name}_W", f"{lyr.name}_b"
        arch_list.append({
            "name": lyr.name,
            "type": "Dense",
            "config": {"activation": lyr.activation.__name__},
            "weights": [W_key, b_key]
        })
        weights_dict[W_key] = W.astype(np.float32)
        weights_dict[b_key] = b.astype(np.float32)
    else:
        raise ValueError(f"Unsupported layer type: {type(lyr)}")

with open(JSON_PATH, "w") as f:
    json.dump(arch_list, f, indent=2)
print(f"Architecture saved → {JSON_PATH}")

np.savez(NPZ_PATH, **weights_dict)
print(f"Weights saved → {NPZ_PATH}")