import argparse
from src.predict import predict_image


def main():
    parser = argparse.ArgumentParser(
        description="Classify a chest X-ray image (educational use only)."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/covid_xray_model.keras")
    parser.add_argument("--labels", default="models/class_names.json")
    args = parser.parse_args()

    label, score = predict_image(args.image, args.model, args.labels)
    print(f"Predicted class: {label}")
    print(f"Model score: {score:.4f}")
    print("Not a medical diagnosis. Consult a qualified healthcare professional.")


if __name__ == "__main__":
    main()
