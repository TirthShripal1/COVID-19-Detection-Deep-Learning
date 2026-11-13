import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from .data_loader import load_dataset


def build_model(num_classes, weights="imagenet"):
    selected_weights = None if weights == "none" else weights
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=selected_weights,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the chest X-ray classifier.")
    parser.add_argument("--data-dir", default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    tf.keras.utils.set_random_seed(args.seed)
    data_root = Path(args.data_dir)
    train_ds, class_names = load_dataset(
        data_root / "train", args.batch_size, shuffle=True, seed=args.seed
    )
    val_ds, val_classes = load_dataset(
        data_root / "val", args.batch_size, shuffle=False, seed=args.seed
    )
    if class_names != val_classes:
        raise ValueError("Train and validation class names/order do not match.")

    model = build_model(len(class_names), args.weights)
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    with open("models/class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/covid_xray_model.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )

    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        plt.figure()
        plt.plot(history.history[metric], label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(f"Training and validation {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"reports/{metric}_curve.png", dpi=150)
        plt.close()

    print("Training finished. Best checkpoint: models/covid_xray_model.keras")
    print("Class order:", class_names)


if __name__ == "__main__":
    main()
