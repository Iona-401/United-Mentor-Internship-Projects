import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import joblib
import numpy as np
import json
import os
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

TF_ENABLE_ONEDNN_OPTS = 0


class TrainingMonitor(Callback):
    """Custom Callback for real time monitoring"""

    def __init__(self):
        super().__init__()
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        # Store training metrics
        for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            if key in logs:
                self.training_history[key].append(logs[key])

        # Store learning rate
        lr = float(self.model.optimizer.learning_rate)
        self.training_history["learning_rate"].append(lr)

        # Print Progress
        print(
            f"📊 Epoch {epoch+1}: "
            f"Loss: {logs['loss']:.4f}, "
            f"Acc: {logs['accuracy']:.4f}, "
            f"Val Loss: {logs['val_loss']:.4f}, "
            f"Val Acc: {logs['val_accuracy']:.4f}, "
            f"LR: {lr:.2e}"
        )


def create_performance_plots(histories, model_names, save_dir=None):
    """Create Comprehensive performance visualization plots"""

    if save_dir is None:
        save_dir = os.path.join(script_dir, "plots")

    os.makedirs(save_dir, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Animal Classification Model Performance Analysis",
        fontsize=16,
        fontweight="bold",
    )

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    # Plot 1: Training & Validation Accuracy
    ax1 = axes[0, 0]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if hasattr(history, "history"):
            epochs = range(1, len(history.history["accuracy"]) + 1)
            ax1.plot(
                epochs,
                history.history["accuracy"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} - Training",
                linestyle="-",
            )
            ax1.plot(
                epochs,
                history.history["val_accuracy"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} - Validation",
                linestyle="--",
                alpha=0.8,
            )

    ax1.set_title("📈 Model Accuracy Over Time", fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Training & Validation Loss
    ax2 = axes[0, 1]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if hasattr(history, "history"):
            epochs = range(1, len(history.history["loss"]) + 1)
            ax2.plot(
                epochs,
                history.history["loss"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} - Training",
                linestyle="-",
            )
            ax2.plot(
                epochs,
                history.history["val_loss"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} - Validation",
                linestyle="--",
                alpha=0.8,
            )

    ax2.set_title("📉 Model Loss Over Time", fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if hasattr(history, "history") and "lr" in history.history:
            epochs = range(1, len(history.history["lr"]) + 1)
            ax3.semilogy(
                epochs,
                history.history["lr"],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} LR",
                marker="o",
                markersize=3,
            )

    ax3.set_title("⚡ Learning Rate Schedule", fontweight="bold")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Learning Rate (log scale)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Model Comparison Bar Chart
    ax4 = axes[1, 1]
    final_accuracies = []
    for history in histories:
        if hasattr(history, "history"):
            final_accuracies.append(max(history.history["val_accuracy"]))
        else:
            final_accuracies.append(0)

    bars = ax4.bar(
        model_names, final_accuracies, color=colors[: len(model_names)], alpha=0.8
    )
    ax4.set_title("🏆 Final Model Performance Comparison", fontweight="bold")
    ax4.set_ylabel("Validation Accuracy")
    ax4.set_ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(save_dir, f"performance_analysis_{timestamp}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(save_dir, "performance_analysis_latest.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"📊 Performance plots saved to {save_dir}")

    plt.show()
    return fig


def save_training_metrics(histories, model_names, save_path=None):
    """Save detailed training metrics for later analysis"""

    if save_path is None:
        save_path = os.path.join(script_dir, "training_metrics.json")

    metrics = {"timestamp": datetime.now().isoformat(), "models": {}}

    for history, name in zip(histories, model_names):
        if hasattr(history, "history"):
            metrics["models"][name] = {
                "final_train_accuracy": float(history.history["accuracy"][-1]),
                "final_val_accuracy": float(history.history["val_accuracy"][-1]),
                "best_val_accuracy": float(max(history.history["val_accuracy"])),
                "final_train_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1]),
                "epochs_trained": len(history.history["accuracy"]),
                "training_history": {
                    "accuracy": [float(x) for x in history.history["accuracy"]],
                    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
                    "loss": [float(x) for x in history.history["loss"]],
                    "val_loss": [float(x) for x in history.history["val_loss"]],
                },
            }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📈 Training metrics saved to {save_path}")


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
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(
    script_dir, "dataset"
)  # Look for dataset in same directory as script
plots_dir = os.path.join(script_dir, "plots")
models_dir = script_dir  # Save models in same directory

img_height = 224
img_width = 224
batch_size = 24

# Check if dataset exists
if not os.path.exists(data_dir):
    print(f"❌ Dataset directory not found: {data_dir}")
    print(f"📁 Current script location: {script_dir}")
    print(f"📁 Looking for dataset at: {data_dir}")
    print("\n💡 Please ensure your dataset is located at:")
    print(f"   {data_dir}")
    print("\nOr update the data_dir path in the script.")
    exit(1)

data_dir_path = pathlib.Path(data_dir)
supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
image_count = 0
for ext in supported_extensions:
    image_count += len(list(data_dir_path.glob(f"*/{ext}")))

print(f"✅ Dataset found: {data_dir}")
print(f"📊 Total images found: {image_count}")


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

if __name__ == "__main__":
    print("🐾 Enhanced Animal Classification Training with Performance Monitoring")
    print("=" * 70)
    print(f"📁 Working directory: {script_dir}")
    print(f"📁 Dataset directory: {data_dir}")

    # Get number of classes
    num_classes = len(class_names)

    # Create model instances
    model_CNN = custom_CNN(num_classes, img_height, img_width)
    model_CNN.compile(
        optimizer=Adam(learning_rate=0.001, weight_decay=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_EF1 = efficientNet(num_classes, img_height, img_width, stage="stage1")
    model_EF1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_EF2 = efficientNet(num_classes, img_height, img_width, stage="stage2")
    model_EF2.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Enhanced Training with Monitoring
    histories = []
    model_names = []

    # Train Custom CNN
    print("\n🔥 Training Custom CNN...")
    cnn_monitor = TrainingMonitor()
    enhanced_CNN_callbacks = CNN_callbacks + [cnn_monitor]

    history_CNN = model_CNN.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=enhanced_CNN_callbacks,
    )

    histories.append(history_CNN)
    model_names.append("Custom CNN")

    # Train EfficientNet Stage 1
    print("\n🚀 Training EfficientNet Stage 1...")
    ef1_monitor = TrainingMonitor()
    enhanced_EF1_callbacks = efficientNet_stage1_callbacks + [ef1_monitor]

    history_EF_stage1 = model_EF1.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=enhanced_EF1_callbacks,
    )

    histories.append(history_EF_stage1)
    model_names.append("EfficientNet Stage 1")

    # Train EfficientNet Stage 2
    print("\n🎯 Training EfficientNet Stage 2...")
    ef2_monitor = TrainingMonitor()
    enhanced_EF2_callbacks = efficientNet_stage2_callbacks + [ef2_monitor]

    history_EF_stage2 = model_EF2.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=enhanced_EF2_callbacks,
    )

    histories.append(history_EF_stage2)
    model_names.append("EfficientNet Stage 2")

    # Create comprehensive performance visualization
    print("\n📊 Creating performance visualizations...")
    create_performance_plots(histories, model_names)

    # Save detailed metrics
    save_training_metrics(histories, model_names)

    # Save models and metadata
    print("\n💾 Saving models...")
    joblib.dump(model_CNN, os.path.join(models_dir, "aniClass_CNN_enhanced.pkl"))
    joblib.dump(model_EF1, os.path.join(models_dir, "aniClass_EFF_Stage1.pkl"))
    joblib.dump(model_EF2, os.path.join(models_dir, "aniClass_EFF_Stage2.pkl"))

    # Save class names for the UI
    with open(os.path.join(models_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    print("✅ Enhanced training completed with performance monitoring!")
    print("📊 Check the plots folder for detailed visualizations!")
    print("📈 Training metrics saved for future analysis!")
