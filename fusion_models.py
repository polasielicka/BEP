import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
emg = np.load("emg_all.npy", allow_pickle=True)
imu = np.load("imu_all.npy", allow_pickle=True)
y = np.load("labels_all.npy", allow_pickle=True).astype(int)

n_classes = len(np.unique(y))

# Transpose to (N, T, C)
emg = np.transpose(emg, (0, 2, 1))  # -> (N, T_emg, 8)
imu = np.transpose(imu, (0, 2, 1))  # -> (N, T_imu, 48)

print("EMG shape:", emg.shape, "IMU shape:", imu.shape)

# Per-segment, per-channel z-score
def zscore_segment(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / sd

emg = zscore_segment(emg)
imu = zscore_segment(imu)

# train val test split using stratified sampling
Xemg_tr, Xemg_tmp, Ximu_tr, Ximu_tmp, y_tr, y_tmp = train_test_split(
    emg, imu, y, test_size=0.30, stratify=y, random_state=SEED
)
Xemg_val, Xemg_te, Ximu_val, Ximu_te, y_val, y_te = train_test_split(
    Xemg_tmp, Ximu_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
)

# shared model backbone
def conv_backbone_block(inp, name_prefix=""):
    x = layers.Conv1D(32, 7, strides=2, padding="same", name=f"{name_prefix}conv1")(inp)
    x = layers.BatchNormalization(name=f"{name_prefix}bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}relu1")(x)

    x = layers.Conv1D(64, 5, strides=2, padding="same", name=f"{name_prefix}conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}relu2")(x)

    x = layers.Conv1D(64, 3, strides=1, padding="same", name=f"{name_prefix}conv3")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}bn3")(x)
    x = layers.ReLU(name=f"{name_prefix}relu3")(x)
    return x

def embedding_head(x, name_prefix=""):
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}gap")(x)
    x = layers.Dense(128, activation="relu", name=f"{name_prefix}fc")(x)
    x = layers.Dropout(0.3, name=f"{name_prefix}drop")(x)
    return x

def classifier_head(x, n_classes, name_prefix=""):
    # returns a probability vector (softmax)
    return layers.Dense(n_classes, activation="softmax", name=f"{name_prefix}softmax")(x)

def compile_model(model):
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_eval(model, X_inputs, y_tr, y_val, y_te, tag):
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                    min_lr=1e-5, verbose=0),
    ]
    hist = model.fit(
        X_inputs["train"], y_tr,
        validation_data=(X_inputs["val"], y_val),
        epochs=50, batch_size=64, verbose=1, callbacks=cb
    )
    y_pred = np.argmax(model.predict(X_inputs["test"], verbose=0), axis=1)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    print(f"{tag} | Test Accuracy: {acc:.3f} | Macro-F1: {f1m:.3f}")
    print(classification_report(y_te, y_pred, digits=3))

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{tag}.h5")
    np.save(f"models/{tag}_y_pred.npy", y_pred)
    np.save(f"models/{tag}_y_test.npy", y_te)
    return acc, f1m

# temporal resize to common length (for early fusion)
def _temporal_resize(target_len: int, name: str, method: str = "bilinear"):
    # Input (B, T, C) -> Output (B, target_len, C)
    return layers.Lambda(
        lambda t: tf.image.resize(
            tf.expand_dims(t, axis=2),
            size=(target_len, 1),
            method=method,
            antialias=False
        )[:, :, 0, :],
        name=name
    )

# early fusion
def build_early_fusion(emg_ch=8, imu_ch=48, target_len=256):
    emg_in = layers.Input(shape=(None, emg_ch), name="emg_in")
    imu_in = layers.Input(shape=(None, imu_ch), name="imu_in")

    # Modality-specific stems
    emg_stem = layers.Conv1D(32, 7, strides=2, padding="same",
                             name="emg_stem_conv")(emg_in)
    emg_stem = layers.BatchNormalization(name="emg_stem_bn")(emg_stem)
    emg_stem = layers.ReLU(name="emg_stem_relu")(emg_stem)

    imu_stem = layers.Conv1D(32, 7, strides=2, padding="same",
                             name="imu_stem_conv")(imu_in)
    imu_stem = layers.BatchNormalization(name="imu_stem_bn")(imu_stem)
    imu_stem = layers.ReLU(name="imu_stem_relu")(imu_stem)

    # Bring to same temporal length (synchronization step)
    emg_resz = _temporal_resize(target_len, name="emg_resize")(emg_stem)
    imu_resz = _temporal_resize(target_len, name="imu_resize")(imu_stem)

    # Feature-level fusion
    fused = layers.Concatenate(axis=-1, name="concat_early")([emg_resz, imu_resz])

    # Shared trunk + classifier
    x = conv_backbone_block(fused, name_prefix="trunk_")
    x = embedding_head(x, name_prefix="trunk_")
    out = classifier_head(x, n_classes, name_prefix="trunk_")

    model = models.Model([emg_in, imu_in], out, name="EarlyFusion")
    return compile_model(model)

# late fusion
def build_late_fusion(emg_ch=8, imu_ch=48):
    emg_in = layers.Input(shape=(None, emg_ch), name="emg_in")
    imu_in = layers.Input(shape=(None, imu_ch), name="imu_in")

    # sEMG branch
    emg_feat = conv_backbone_block(emg_in, name_prefix="emg_")
    emg_emb  = embedding_head(emg_feat, name_prefix="emg_")
    emg_dec  = classifier_head(emg_emb, n_classes, name_prefix="emg_dec_")

    # IMU branch
    imu_feat = conv_backbone_block(imu_in, name_prefix="imu_")
    imu_emb  = embedding_head(imu_feat, name_prefix="imu_")
    imu_dec  = classifier_head(imu_emb, n_classes, name_prefix="imu_dec_")

    # Decision-level fusion (average of modality decisions)
    fused_dec = layers.Average(name="avg_decisions")([emg_dec, imu_dec])

    model = models.Model([emg_in, imu_in], fused_dec, name="LateFusion")
    return compile_model(model)

# hybrid fusion
def build_hybrid_fusion(emg_ch=8, imu_ch=48):
    emg_in = layers.Input(shape=(None, emg_ch), name="emg_in")
    imu_in = layers.Input(shape=(None, imu_ch), name="imu_in")

    # Modality-specific backbones
    emg_feat = conv_backbone_block(emg_in, name_prefix="emg_")
    imu_feat = conv_backbone_block(imu_in, name_prefix="imu_")

    # Embeddings
    emg_emb = embedding_head(emg_feat, name_prefix="emg_")
    imu_emb = embedding_head(imu_feat, name_prefix="imu_")

    # Unimodal decision heads (decision-level part)
    emg_dec = classifier_head(emg_emb, n_classes, name_prefix="hy_emg_dec_")
    imu_dec = classifier_head(imu_emb, n_classes, name_prefix="hy_imu_dec_")

    # Feature-level fusion branch (embedding concatenation)
    fused_emb = layers.Concatenate(name="concat_embeddings")([emg_emb, imu_emb])
    x = layers.Dense(128, activation="relu", name="fusion_fc")(fused_emb)
    x = layers.Dropout(0.3, name="fusion_drop")(x)
    fusion_dec = classifier_head(x, n_classes, name_prefix="hy_fusion_")

    # Hybrid fusion of decisions: combine unimodal + fused decision vectors
    out = layers.Average(name="avg_hybrid_decisions")([emg_dec, imu_dec, fusion_dec])

    model = models.Model([emg_in, imu_in], out, name="HybridFusion")
    return compile_model(model)

early = build_early_fusion()
late  = build_late_fusion()
hyb   = build_hybrid_fusion()

def pack(Xemg_a, Ximu_a, Xemg_b, Ximu_b, Xemg_c, Ximu_c):
    return {
        "train": [Xemg_a, Ximu_a],
        "val":   [Xemg_b, Ximu_b],
        "test":  [Xemg_c, Ximu_c],
    }

inputs = pack(Xemg_tr, Ximu_tr, Xemg_val, Ximu_val, Xemg_te, Ximu_te)

# training and evaluation
acc_e, f1_e = train_eval(early, inputs, y_tr, y_val, y_te, tag="fusion_early")
acc_l, f1_l = train_eval(late,  inputs, y_tr, y_val, y_te, tag="fusion_late")
acc_h, f1_h = train_eval(hyb,   inputs, y_tr, y_val, y_te, tag="fusion_hybrid")

print("\nSUMMARY:")
print(f"Early  Fusion | ACC={acc_e:.3f} | F1m={f1_e:.3f}")
print(f"Late   Fusion | ACC={acc_l:.3f} | F1m={f1_l:.3f}")
print(f"Hybrid Fusion | ACC={acc_h:.3f} | F1m={f1_h:.3f}")
