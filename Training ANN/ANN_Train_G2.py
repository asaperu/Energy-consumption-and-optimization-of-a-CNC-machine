import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ======================================================
# PARAMETRI
# ======================================================
EXCEL_FILE = "Dati_allenamento.xlsx"
SHEET_NAME = "G2"

FEATURES = ["R", "F", "S", "MRR"]
TARGET = "P"

# ======================================================
# 1. LETTURA DATI
# ======================================================
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

X = df[FEATURES].values
Y = df[TARGET].values.reshape(-1, 1)   # P in Watt

# ======================================================
# 2. TRAIN / TEST SPLIT (PRIMA DI SCALARE)
# ======================================================
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    train_size=0.8,
    random_state=42
)

# ======================================================
# 3. SCALER
# ======================================================
# Input
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled  = x_scaler.transform(x_test)

# Target
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled  = y_scaler.transform(y_test)

joblib.dump(x_scaler, "scalerX2_G2.pkl")
joblib.dump(y_scaler, "scalerY2_G2.pkl")

# ======================================================
# 4. MODELLO ANN
# ======================================================
l2_reg = keras.regularizers.l2(1e-3)

model = keras.Sequential([
    keras.layers.Dense(
        10,
        kernel_regularizer=l2_reg,
        input_shape=(x_train_scaled.shape[1],)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    #keras.layers.Dropout(0.2),

    keras.layers.Dense(20, kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Dense(50, kernel_regularizer=l2_reg),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    #keras.layers.Dropout(0.2),


    keras.layers.Dense(100, kernel_regularizer=l2_reg),
    keras.layers.ReLU(),

    keras.layers.Dense(1)   # output: P scalata
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",      # MSE su P scalata
    metrics=["mae"]  # MAE su P scalata
)

# ======================================================
# 5. CALLBACKS
# ======================================================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=False
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath="checkpoint_G2.best.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=0
)

# ======================================================
# 6. TRAINING
# ======================================================
history = model.fit(
    x_train_scaled,
    y_train_scaled,
    epochs=300,
    batch_size=30,
    validation_data=(x_test_scaled, y_test_scaled),
    callbacks=[early_stop, lr_scheduler, checkpoint_cb],
    verbose=1
)

# ======================================================
# 7. RIPRISTINO BEST MODEL
# ======================================================
model.load_weights("checkpoint_G2.best.weights.h5")

# ======================================================
# ======================= PLOT =========================
# ======================================================

# -----------------------------
# LOSS (spazio scalato)
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss G2 (MSE – spazio scalato)")
plt.xlabel("Epoche")
plt.ylabel("MSE (scaled)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# MAE (spazio scalato)
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("MAE G2 (spazio scalato)")
plt.xlabel("Epoche")
plt.ylabel("MAE (scaled)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
# 8. SCATTER P REALE vs P PREDETTA (WATT)
# ======================================================
y_test_pred_scaled = model.predict(x_test_scaled)
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_test_pred, alpha=0.6)

min_val = float(min(y_test.min(), y_test_pred.min()))
max_val = float(max(y_test.max(), y_test_pred.max()))
plt.plot([min_val, max_val], [min_val, max_val], "k--")

plt.xlabel("P reale [W]")
plt.ylabel("P predetta [W]")
plt.title("P reale vs P predetta – G2")
plt.grid(True)
plt.tight_layout()
plt.show()
