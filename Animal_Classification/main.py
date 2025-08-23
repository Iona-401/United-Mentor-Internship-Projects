import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import joblib

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

TF_ENABLE_ONEDNN_OPTS = 0


# MODEL DEFINITION
def custom_CNN(num_classes, img_height, img_width):
    model = Sequential(
        [
            layers.Input(shape=(img_height, img_width, 3)),
            # Enhanced Data Augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomZoom(0.2),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
            # First Conv Block
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            # Second Conv Block
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.15),
            # Third Conv Block
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            # Fourth Conv Block
            layers.Conv2D(512, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            # Fifth Conv Block
            layers.Conv2D(512, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            # Enhanced classifier
            layers.Dropout(0.4),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def efficientNet(num_classes, img_height, img_width, stage):
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3),
    )

    if stage == "stage1":
        # Freeze Base Model for initial train
        base_model.trainable = False
        print("EfficientNetB0 Frozen for Stage 1 training")
    elif stage == "stage2":
        # Unfreeze for fine tuning
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        print("EfficientNetB0 unfrozen for Stage 2 fine-tuning")
    model = Sequential(
        [
            layers.Input(shape=(img_height, img_width, 3)),
            # Data Augmentation
            # layers.RandomFlip("horizontal"),
            # layers.RandomRotation(0.15),
            # layers.RandomZoom(0.1),
            # layers.RandomContrast(0.1),
            # Preprocessing for EfficientNet
            base_model,
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# DATA HANDLING
# Data directory setup
data_dir = "Animal Classification/dataset"
img_height = 224
img_width = 224
batch_size = 24

data_dir_path = pathlib.Path(data_dir)
supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
image_count = 0
for ext in supported_extensions:
    image_count += len(list(data_dir_path.glob(f"*/{ext}")))

print(f"Total images found: {image_count}")

print("\nClass distribution:")
for class_dir in data_dir_path.iterdir():
    if class_dir.is_dir():
        class_images = 0
        for ext in supported_extensions:
            class_images += len(list(class_dir.glob(ext)))
            class_images += len(list(class_dir.glob(ext.upper())))
        print(f"{class_dir.name}: {class_images} images")

print("\nSplitting Data")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

print("\nData split into training and validation sets")
print(f"Training set size: {len(train_ds)} batches")
print(f"Validation set size: {len(val_ds)} batches")

class_names = train_ds.class_names
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")

AUTOTUNE = tf.data.AUTOTUNE


def prep_data(ds):
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = prep_data(train_ds)
val_ds = prep_data(val_ds)

train_size = int(1944 * 0.8)
val_size = int(1944 * 0.2)
steps_per_epoch = train_size // batch_size
validation_steps = val_size // batch_size

print(f"Training images: {train_size}")
print(f"Validation images: {val_size}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Callbacks for custom CNN models (more aggressive since training from scratch)
CNN_callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=8,  # More patience for CNNs training from scratch
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,  # More aggressive LR reduction
        patience=4,
        min_lr=1e-8,
        verbose=1,
    ),
]

# Callbacks for transfer learning Stage 1 (frozen backbone)
efficientNet_stage1_callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,  # Less patience since head learns quickly
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,  # Less aggressive since only head is training
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# Callbacks for transfer learning Stage 2 (fine-tuning)
efficientNet_stage2_callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=10,  # More patience for fine-tuning
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # Conservative LR reduction for fine-tuning
        patience=5,
        min_lr=1e-9,  # Lower minimum LR
        verbose=1,
    ),
]
num_classes = len(class_names)

model_CNN = custom_CNN(num_classes, img_height, img_width)
model_CNN.compile(
    optimizer=Adam(learning_rate=0.001, weight_decay=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# history_CNN = model_CNN.fit(
#    train_ds,
#    validation_data=val_ds,
#    epochs=30,
#    verbose=1,
#    steps_per_epoch=steps_per_epoch,
#    validation_steps=validation_steps,
#    callbacks=CNN_callbacks,
# )
# CNN_acc = max(history_CNN.history["val_accuracy"])
# print(f"\nEnhanced CNN Model Training completed! Validation accuracy: {CNN_acc:.4f}")

model_EF1 = efficientNet(num_classes, img_height, img_width, stage="stage1")
model_EF1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_EF_stage1 = model_EF1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=efficientNet_stage1_callbacks,
)
efficientNet_stage1_best_acc = max(history_EF_stage1.history["val_accuracy"])
print(
    f"\nEfficientNet Stage 1 Training completed! Validation accuracy: {efficientNet_stage1_best_acc:.4f}"
)

model_EF2 = efficientNet(num_classes, img_height, img_width, stage="stage2")
model_EF2.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_EF_stage2 = model_EF2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=efficientNet_stage2_callbacks,
)
efficientNet_stage2_best_acc = max(history_EF_stage2.history["val_accuracy"])
print(
    f"\nEfficientNet Stage 2 Training completed! Validation accuracy: {efficientNet_stage2_best_acc:.4f}"
)

# Save class names for the UI
import json

with open("Animal Classification/class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved for UI")

# Saving Models
joblib.dump(model_EF1, "Animal Classification/aniClass_EFF_Stage1.pkl")
joblib.dump(model_EF2, "Animal Classification/aniClass_EFF_Stage2.pkl")
print("Models saved successfully!")
