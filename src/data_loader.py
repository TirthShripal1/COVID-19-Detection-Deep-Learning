from pathlib import Path
import tensorflow as tf
from .preprocessing import IMAGE_SIZE, preprocess_image

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")


def validate_split_directory(directory):
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    class_dirs = sorted(p for p in path.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(f"Expected at least two class subfolders inside {path}.")
    count = sum(
        1 for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
    )
    if count == 0:
        raise ValueError(f"No supported images found in {path}")
    return [p.name for p in class_dirs]


def load_dataset(directory, batch_size=32, shuffle=False, seed=42):
    class_names = validate_split_directory(directory)
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    dataset = dataset.map(
        lambda images, labels: (preprocess_image(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE), class_names
