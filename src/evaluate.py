import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from .data_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained classifier.")
    parser.add_argument("--data-dir", default="data/chest_xray/test")
    parser.add_argument("--model", default="models/covid_xray_model.keras")
    parser.add_argument("--labels", default="models/class_names.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model not found: {args.model}. Train first.")
    with open(args.labels, encoding="utf-8") as file:
        class_names = json.load(file)

    dataset, found_classes = load_dataset(args.data_dir, args.batch_size)
    if class_names != found_classes:
        raise ValueError("Test class names/order differ from saved training labels.")

    model = tf.keras.models.load_model(args.model)
    loss, accuracy = model.evaluate(dataset, verbose=1)
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    probabilities = model.predict(dataset, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))))


if __name__ == "__main__":
    main()
