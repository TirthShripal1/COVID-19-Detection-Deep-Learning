import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .preprocessing import IMAGE_SIZE


def predict_image(image_path, model_path="models/covid_xray_model.keras",
                  labels_path="models/class_names.json"):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

    with open(labels_path, encoding="utf-8") as file:
        class_names = json.load(file)

    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])


def main():
    parser = argparse.ArgumentParser(description="Predict one X-ray image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/covid_xray_model.keras")
    parser.add_argument("--labels", default="models/class_names.json")
    args = parser.parse_args()
    label, score = predict_image(args.image, args.model, args.labels)
    print(f"Predicted class: {label}")
    print(f"Model score: {score:.4f}")
    print("Educational output only; not a medical diagnosis.")


if __name__ == "__main__":
    main()
