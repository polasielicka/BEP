import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os, random

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
emg = np.load("emg_all.npy", allow_pickle=True)     # (N, 8, 5037)
imu = np.load("imu_all.npy", allow_pickle=True)     # (N, 48, 592)
labels = np.load("labels_all.npy", allow_pickle=True).astype(int)
subjects = np.load("subjects_all.npy", allow_pickle=True).astype(int)

# Change to imu or emg
X = imu
modality = "imu"

# Transpose to (N, T, C)
X = np.transpose(X, (0, 2, 1))
n_classes = len(np.unique(labels))

# z-score normalization
def zscore_segment(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd

X = zscore_segment(X)

# train val test split using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.3, stratify=labels, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

# model architecture
def build_1d_cnn(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, strides=2, padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_1d_cnn(X_train.shape[1:], n_classes)
model.summary()

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=cb
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.3f}")

# Save model and arrays for later in models folder
os.makedirs("models", exist_ok=True)
model.save(f"models/{modality}_1dcnn.h5")

np.save(f"models/{modality}_X_test.npy", X_test)
np.save(f"models/{modality}_y_test.npy", y_test)

# Save predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
np.save(f"models/{modality}_y_pred.npy", y_pred)
